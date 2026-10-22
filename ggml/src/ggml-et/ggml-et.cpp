#समावेश "ggml-et.h"

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-et-common.h"
#समावेश "ggml-et-kernels.h"
#समावेश "ggml-et-memops.h"
#समावेश "ggml-et-ops.h"
#समावेश "ggml-impl.h"
#समावेश "ggml.h"

#समावेश <stdarg.h>

#समावेश <cstdarg>
#समावेश <cstdio>
#समावेश <cstdlib>
#समावेश <cstring>
#समावेश <string>
#समावेश <vector>

#अगर __has_include(<filesystem>)
#    include <filesystem>
namespace fs = std::filesystem;
#या_अगर __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#अन्यथा
#    error "cannot include the filesystem library"
#पूर्ण_अगर

/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
अटल व्योम ggml_et_dump_tensor_metadata(स्थिर ggml_tensor * ggtensor, माप_प्रकार indent_level, स्थिर अक्षर * title) {
    std::string spaces(indent_level, ' ');
    ख_लिखो(मानक_त्रुटि,
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
            spaces.c_str(), ggtensor->तथ्य, spaces.c_str(), (व्योम *) ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
अटल व्योम ggml_et_dump_operator_metadata(स्थिर ggml_tensor * ggtensor) {
    GGML_ASSERT(ggtensor != शून्य);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC && ggtensor->src[i]; i++) {
        अक्षर arr[16];
        पूर्णांक  n = snprintf(arr, माप(arr), "src[%i]->name", i);
        GGML_ASSERT((अचिन्हित) n < माप(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    अगर (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

अटल काष्ठा ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::बख_प्रवाह>                    profile_stream;
    std::unique_ptr<std::बख_प्रवाह>                    kernel_id_stream;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    bool                                              profiling_enabled = false;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
अटल std::string ggml_et_get_default_et_path() {
    // List of environment variables to check in order of preference
    स्थिर अक्षर * स्थिर env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    क्रम (स्थिर अक्षर * var : env_vars) {
        अगर (स्थिर अक्षर * et_path = std::दो_पर्या(var)) {
            अगर (et_path && *et_path != '\0') {
                वापस fs::path(et_path).string();
            }
        }
    }

    // Otherwise assume default
    वापस fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
अटल अंतरभूत स्वतः ggml_et_get_default_sysemu_options() {
    constexpr uint64_t kSysEmuMaxCycles        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t kSysEmuMinionShiresMask = 0x1FFFFFFFFu;
    स्थिर std::string  et_path                 = ggml_et_get_default_et_path() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootromTrampolineToBL2ElfPath =
        et_path + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.spBL2ElfPath =
        et_path + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machineMinionElfPath = et_path + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.masterMinionElfPath  = et_path + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.workerMinionElfPath  = et_path + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executablePath       = et_path + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    स्थिर std::vector<std::string> required_files = {
        sysEmuOptions.bootromTrampolineToBL2ElfPath, sysEmuOptions.spBL2ElfPath,
        sysEmuOptions.machineMinionElfPath,          sysEmuOptions.masterMinionElfPath,
        sysEmuOptions.workerMinionElfPath,           sysEmuOptions.executablePath,
    };

    क्रम (स्थिर स्वतः & file : required_files) {
        अगर (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            पात();
        }
    }

    sysEmuOptions.runDir           = (fs::current_path().string() + "/");
    sysEmuOptions.maxCycles        = kSysEmuMaxCycles;
    sysEmuOptions.minionShiresMask = kSysEmuMinionShiresMask;
    sysEmuOptions.puUart0Path      = sysEmuOptions.runDir + "pu_uart0_tx.log";
    sysEmuOptions.puUart1Path      = sysEmuOptions.runDir + "pu_uart1_tx.log";
    sysEmuOptions.spUart0Path      = sysEmuOptions.runDir + "spio_uart0_tx.log";
    sysEmuOptions.spUart1Path      = sysEmuOptions.runDir + "spio_uart1_tx.log";
    sysEmuOptions.startGdb         = false;
    sysEmuOptions.memcheck         = false;

    वापस sysEmuOptions;
}

// Forward declaration
अटल व्योम ggml_et_driver_cleanup();

अटल bool ggml_et_driver_init() {
    अगर (_drv.runtime != nullptr) {
        निश्चित(_drv.device_layer != nullptr);
    } अन्यथा {
        प्रयास {
#अगर defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#अन्यथा
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#पूर्ण_अगर  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            स्थिर अक्षर * profile_path = दो_पर्या("GGML_ET_PROFILE");
            अगर (profile_path) {
                std::string output_path    = std::string(profile_path) + "/et_runtime_trace.json";
                std::string kernel_id_path = std::string(profile_path) + "/kernel_id.json";

                _drv.profile_stream   = std::make_unique<std::बख_प्रवाह>(output_path);
                _drv.kernel_id_stream = std::make_unique<std::बख_प्रवाह>(kernel_id_path);
                अगर (!_drv.profile_stream->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_path.c_str());
                    पात();
                }
                अगर (!_drv.kernel_id_stream->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_path.c_str());
                    पात();
                }

                स्वतः * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_stream, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = true;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::निकास_पर(ggml_et_driver_cleanup);
            }
        } पकड़ो (स्थिर std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            अगर (_drv.device_layer != nullptr) {
                _drv.device_layer.reset();
            }
            अगर (_drv.runtime != nullptr) {
                _drv.runtime.reset();
            }
            वापस false;
        }
    }
    वापस true;
}

अटल std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    वापस _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    वापस _drv.runtime;
}

अटल व्योम ggml_et_driver_cleanup() {
    अगर (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        स्वतः * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = false;

        अगर (_drv.profile_stream) {
            _drv.profile_stream->बंद();
            _drv.profile_stream.reset();
        }

        // Save kernel map
        अगर (_drv.kernel_id_stream && !_drv.kernel_map.empty()) {
            स्वतः & os = *_drv.kernel_id_stream;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            क्रम (माप_प्रकार i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].first << "\": " << (पूर्णांक) _drv.kernel_map[i].second;
                अगर (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_stream->बंद();
            _drv.kernel_id_stream.reset();
        }
    }
}

अटल ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार devidx);

अटल व्योम ggml_backend_et_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_et_बफर_context * ctx = (ggml_backend_et_बफर_context *) बफर->context;
    अगर (ctx->तथ्य != nullptr) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        अगर (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->तथ्य));
        }
    }
    मिटाओ ctx;
}

अटल व्योम * ggml_backend_et_बफर_get_base(ggml_backend_बफर_t बफर) {
    ggml_backend_et_बफर_context * ctx = (ggml_backend_et_बफर_context *) बफर->context;
    वापस ctx->तथ्य;
}

अटल ggml_status ggml_backend_et_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    अगर (tensor->view_src != शून्य) {
        वापस GGML_STATUS_SUCCESS;
    }

    स्थिर माप_प्रकार original_आकार = ggml_nbytes(tensor);
    स्थिर माप_प्रकार padded_आकार   = ggml_backend_buft_get_alloc_आकार(बफर->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    अगर (padded_आकार > original_आकार) {
        स्थिर माप_प्रकार padding_आकार = padded_आकार - original_आकार;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) बफर->buft->device->context;
        अगर (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            वापस GGML_STATUS_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->तथ्य) + original_आकार;
        अगर (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_आकार)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            वापस GGML_STATUS_FAILED;
        }
    }

    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_et_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                              ggml_tensor *         tensor,
                                              स्थिर व्योम *          तथ्य,
                                              माप_प्रकार                offset,
                                              माप_प्रकार                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        वापस;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) बफर->buft->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->तथ्य) + offset;
    स्थिर std::byte * src_ptr = static_cast<स्थिर std::byte *>(तथ्य);

    rt::EventId event = runtime->memcpyHostToDevice(stream, src_ptr, dst_ptr, size, true /*barrier*/);

    runtime->waitForEvent(event);
}

अटल व्योम ggml_backend_et_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                              स्थिर ggml_tensor *   tensor,
                                              व्योम *                तथ्य,
                                              माप_प्रकार                offset,
                                              माप_प्रकार                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        वापस;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) बफर->buft->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    स्थिर std::byte * src_ptr = static_cast<स्थिर std::byte *>(tensor->तथ्य) + offset;
    std::byte *       dst_ptr = static_cast<std::byte *>(तथ्य);

    rt::EventId event = runtime->memcpyDeviceToHost(stream, src_ptr, dst_ptr, size, true /*barrier*/);

    runtime->waitForEvent(event);
}

अटल bool ggml_backend_et_बफर_cpy_tensor(ggml_backend_बफर_t बफर,
                                              स्थिर ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_UNUSED(बफर);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    वापस false;
}

अटल व्योम ggml_backend_et_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_et_बफर_context * ctx = (ggml_backend_et_बफर_context *) बफर->context;

    अगर (ctx->size == 0 || ctx->तथ्य == nullptr) {
        वापस;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) बफर->buft->device->context;
    अगर (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        वापस;
    }

    // Use device-side memset kernel for efficient clearing
    अगर (!ggml_et_memset(dev_ctx, ctx->तथ्य, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        वापस;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

अटल स्थिर काष्ठा ggml_backend_बफर_i ggml_backend_et_बफर_i = {
    /* .free_buffer     = */ ggml_backend_et_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_et_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_et_बफर_init_tensor,
    /* .memset_tensor   = */ शून्य,
    /* .set_tensor      = */ ggml_backend_et_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_et_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ ggml_backend_et_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_et_बफर_clear,
    /* .reset           = */ शून्य,
};

अटल स्थिर अक्षर * ggml_backend_et_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    GGML_UNUSED(buft);
    वापस GGML_ET_NAME;
}

अटल ggml_backend_बफर_t ggml_backend_et_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    ggml_backend_et_बफर_type_context * btctx = (ggml_backend_et_बफर_type_context *) buft->context;

    ggml_backend_et_बफर_context * ctx = नया ggml_backend_et_बफर_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        मिटाओ ctx;
        वापस nullptr;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    अगर (static_cast<माप_प्रकार>(btctx->devidx) >= rtids.size()) {
        मिटाओ ctx;
        वापस nullptr;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->तथ्य = runtime->mallocDevice(ctx->rtid, size);
    अगर (ctx->तथ्य == nullptr) {
        मिटाओ ctx;
        वापस nullptr;
    }

    वापस ggml_backend_बफर_init(buft, ggml_backend_et_बफर_i, ctx, size);
}

अटल माप_प्रकार ggml_backend_et_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime || !buft->device) {
        वापस GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    वापस prop.cache_पंक्ति_आकार;
}

अटल माप_प्रकार ggml_backend_et_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    अगर (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        वापस dev_ctx->योग_mem;
    }
    वापस SIZE_MAX;
}

अटल माप_प्रकार ggml_backend_et_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    वापस ggml_nbytes_pad(tensor);
}

अटल bool ggml_backend_et_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    GGML_UNUSED(buft);
    वापस false;
}

अटल स्थिर काष्ठा ggml_backend_बफर_type_i ggml_backend_et_बफर_type_i = {
    /* .get_name            = */ ggml_backend_et_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ शून्य,
    /* .get_alignment       = */ ggml_backend_et_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_et_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ ggml_backend_et_बफर_type_is_host,
};

अटल स्थिर अक्षर * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    वापस GGML_ET_NAME;
}

अटल व्योम ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context *     et_ctx  = (ggml_backend_et_context *) backend->context;
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    अगर (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;

        अगर (_drv.profiling_enabled) {
            स्वतः kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        अगर (runtime) {
            अगर (dev_ctx->trace_बफर) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_बफर);
                dev_ctx->trace_बफर = nullptr;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->waitForStream(dev_ctx->default_stream);
            क्रम (स्वतः & slot : dev_ctx->uberkernel.slots) {
                अगर (slot.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
                    slot.device_insts = nullptr;
                }
                अगर (slot.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, slot.device_params);
                    slot.device_params = nullptr;
                }
                slot.has_pending = false;
            }
        }
    }

    मिटाओ et_ctx;
    मिटाओ backend;
}

अटल ggml_backend_बफर_type_t ggml_backend_et_get_default_बफर_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    वापस ggml_backend_et_बफर_type(et_ctx->devidx);
}

अटल व्योम ggml_backend_et_set_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             स्थिर व्योम *   तथ्य,
                                             माप_प्रकार         offset,
                                             माप_प्रकार         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        वापस;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->तथ्य) + offset;
    स्थिर std::byte * src_ptr = static_cast<स्थिर std::byte *>(तथ्य);

    runtime->memcpyHostToDevice(stream, src_ptr, dst_ptr, size, true /*barrier*/);
}

अटल व्योम ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             स्थिर ggml_tensor * tensor,
                                             व्योम *              तथ्य,
                                             माप_प्रकार              offset,
                                             माप_प्रकार              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        वापस;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    स्थिर std::byte * src_ptr = static_cast<स्थिर std::byte *>(tensor->तथ्य) + offset;
    std::byte *       dst_ptr = static_cast<std::byte *>(तथ्य);

    runtime->memcpyDeviceToHost(stream, src_ptr, dst_ptr, size, true /*barrier*/);
}

अटल bool ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             स्थिर ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    वापस false;
}

अटल व्योम ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        वापस;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->waitForStream(dev_ctx->default_stream);

    स्वतः errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
    अगर (errors.empty()) {
        वापस;
    }
    क्रम (स्थिर स्वतः & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (पूर्णांक) err.errorCode_,
                       (पूर्णांक) err.errorContext_.value()[0].type_);
    }
    पात();
}

अटल bool ggml_et_can_fuse(स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, std::initializer_list<ggml_op> ops) {
    अगर (!ggml_can_fuse(cgraph, गाँठ_idx, ops)) {
        वापस false;
    }

    अगर (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        स्थिर ggml_tensor * mm  = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * add = cgraph->nodes[गाँठ_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        अगर (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            वापस false;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        अगर (add->type != GGML_TYPE_F32) {
            वापस false;
        }
        अगर (add->src[0] != mm && add->src[1] != mm) {
            वापस false;
        }

        स्थिर ggml_tensor * bias = (add->src[0] == mm) ? add->src[1] : add->src[0];

        अगर (bias->type != GGML_TYPE_F32) {
            वापस false;
        }

        // No broadcasting: bias shape must equal MM output shape.
        क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
            अगर (bias->ne[i] != mm->ne[i]) {
                वापस false;
            }
        }

        // Bias and dst must be contiguous and have identical strides - the
        // kernel uses dst-style offset arithmetic against bias's nb[].
        अगर (!ggml_is_contiguous(bias) || !ggml_is_contiguous(mm)) {
            वापस false;
        }
        क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
            अगर ((int64_t) bias->nb[i] != (int64_t) add->nb[i]) {
                वापस false;
            }
        }
    }

    अगर (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        स्थिर ggml_tensor * rms_norm = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * mul      = cgraph->nodes[गाँठ_idx + 1];

        // ET only supports F32
        अगर (rms_norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            वापस false;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        स्थिर ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        अगर (weights->type != GGML_TYPE_F32) {
            वापस false;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        अगर (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_rows(weights)) {
            वापस false;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        अगर (rms_norm->src[0]->ne[0] % 16 != 0 || weights->ne[0] % 16 != 0) {
            वापस false;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        अगर (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            वापस false;
        }
    }

    वापस true;
}

अटल ggml_status ggml_backend_et_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_graph(&dev_ctx->uberkernel);

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाँठ = cgraph->nodes[i];

        अगर (गाँठ->op == GGML_OP_NONE || गाँठ->op == GGML_OP_VIEW || गाँठ->op == GGML_OP_RESHAPE ||
            गाँठ->op == GGML_OP_PERMUTE || गाँठ->op == GGML_OP_TRANSPOSE) {
            जारी;
        }

        // --- Fusion checks (before regular dispatch) ---
        अगर (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, गाँठ, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            जारी;
        }
        अगर (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
            ggml_et_op_mul_mat(dev_ctx, गाँठ, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            जारी;
        }

        चयन (गाँठ->op) {
            हाल GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_योग_ROWS:
                ggml_et_op_योग_rows(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_MEAN:
                ggml_et_op_mean(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_ADD:
                ggml_et_op_add(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_GROUP_NORM:
                ggml_et_op_group_norm(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_IM2COL:
                ggml_et_op_im2col(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_GET_ROWS:
                ggml_et_op_get_rows(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_SSM_SCAN:
                ggml_et_op_ssm_scan(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_SET_ROWS:
                ggml_et_op_set_rows(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_SET:
                ggml_et_op_set(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, गाँठ);
                अवरोध;

            हाल GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, गाँठ);
                अवरोध;

            शेष:
                ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(गाँठ->op));
                वापस GGML_STATUS_FAILED;
        }

        अगर (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
            वापस GGML_STATUS_FAILED;
        }
    }

    अगर (!ggml_et_uberkernel_end_graph(dev_ctx)) {
        ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
        वापस GGML_STATUS_FAILED;
    }

    वापस GGML_STATUS_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
अटल bool et_ggml_is_row_contiguous(स्थिर ggml_tensor * t) {
    वापस t->nb[0] == ggml_type_आकार(t->type);
}

अटल bool ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    GGML_UNUSED(dev);

    bool supported = false;
    चयन (op->op) {
        हाल GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == माप(भग्न) && ggml_is_contiguous(op);
            अवरोध;
        हाल GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_योग_ROWS:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_MEAN:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == माप(भग्न) && op->src[0]->nb[0] == माप(भग्न)) {
                चयन (ggml_get_unary_op(op)) {
                    हाल GGML_UNARY_OP_ABS:
                    हाल GGML_UNARY_OP_SGN:
                    हाल GGML_UNARY_OP_NEG:
                    हाल GGML_UNARY_OP_STEP:
                    हाल GGML_UNARY_OP_TANH:
                    हाल GGML_UNARY_OP_ELU:
                    हाल GGML_UNARY_OP_RELU:
                    हाल GGML_UNARY_OP_SIGMOID:
                    हाल GGML_UNARY_OP_GELU:
                    हाल GGML_UNARY_OP_GELU_QUICK:
                    हाल GGML_UNARY_OP_SILU:
                    हाल GGML_UNARY_OP_HARDSWISH:
                    हाल GGML_UNARY_OP_HARDSIGMOID:
                    हाल GGML_UNARY_OP_EXP:
                    हाल GGML_UNARY_OP_EXPM1:
                    हाल GGML_UNARY_OP_SOFTPLUS:
                    हाल GGML_UNARY_OP_GELU_ERF:
                    हाल GGML_UNARY_OP_FLOOR:
                    हाल GGML_UNARY_OP_CEIL:
                    हाल GGML_UNARY_OP_ROUND:
                    हाल GGML_UNARY_OP_TRUNC:
                        supported = true;
                        अवरोध;
                    शेष:
                        अवरोध;
                }
            }
            अवरोध;
        हाल GGML_OP_MUL:
        हाल GGML_OP_ADD:
        हाल GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->nb[0] == माप(भग्न) &&
                        op->src[0]->nb[0] == माप(भग्न) &&
                        (op->src[1]->nb[0] == माप(भग्न) || op->src[1]->ne[0] == 1) &&
                        op->nb[1] == op->ne[0] * माप(भग्न);
            अवरोध;
        हाल GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            अगर (op->type == GGML_TYPE_F32 &&
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
                supported = true;
            } अन्यथा अगर (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                bool src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));
                bool src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type));
                bool dst_first_dim_contiguous  = (op->nb[0] == माप(भग्न));

                // Check destination stride ordering (only for dimensions with ne > 1)
                bool dst_properly_ordered = true;
                क्रम (पूर्णांक d = 0; d < 3; d++) {
                    अगर (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = false;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;
            } अन्यथा अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                bool src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));
                bool src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type));
                bool dst_first_dim_contiguous  = (op->nb[0] == माप(भग्न));

                bool dst_properly_ordered = true;
                क्रम (पूर्णांक d = 0; d < 3; d++) {
                    अगर (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = false;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;

            } अन्यथा अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                bool src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));
                bool src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type));
                bool dst_first_dim_contiguous  = (op->nb[0] == माप(भग्न));

                bool dst_properly_ordered = true;
                क्रम (पूर्णांक d = 0; d < 3; d++) {
                    अगर (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = false;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/Q4_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            अगर (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_I32) {
                // Check first dimension contiguity requirements (matching CPU backend)
                bool src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));
                bool src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type));
                bool src2_first_dim_contiguous = (op->src[2]->nb[0] == ggml_type_आकार(op->src[2]->type));
                bool dst_first_dim_contiguous  = (op->nb[0] == माप(भग्न));

                // Check destination stride ordering (only for dimensions with ne > 1)
                bool dst_properly_ordered = true;
                क्रम (पूर्णांक d = 0; d < 3; d++) {
                    अगर (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = false;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                bool dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                  (op->src[1]->ne[3] == 1) &&  // b is 3d
                                  (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                  (op->src[2]->ne[1] == op->src[1]->ne[2]) &&    // must have expert list per b row
                                  (op->src[0]->ne[0] == op->src[1]->ne[0]) &&    // K dimension must match
                                  (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && src2_first_dim_contiguous &&
                            dst_first_dim_contiguous && dst_properly_ordered && dims_valid;
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0])) {
                स्थिर पूर्णांक  mode             = ggml_get_op_params_i32(op, 2);
                स्थिर पूर्णांक  ndims            = ggml_get_op_params_i32(op, 1);
                स्थिर bool is_normal        = mode == GGML_ROPE_TYPE_NORMAL;
                स्थिर bool is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                स्थिर bool is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                स्थिर bool zero_view_offset = op->src[0]->view_src == nullptr || op->src[0]->view_offs == 0;
                स्थिर bool has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                स्थिर bool zero_rot_offset  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_offset && zero_rot_offset && ndims <= 512 &&
                    (is_normal || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_GROUP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]) &&
                        ggml_get_op_params_i32(op, 0) > 0;
            अवरोध;
        हाल GGML_OP_IM2COL:
            supported = op->src[0] && op->src[1] &&
                        ((op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                         (op->type == GGML_TYPE_F16 &&
                          (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32))) &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) &&
                        op->nb[0] == ggml_type_आकार(op->type) && op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type);
            अवरोध;
        हाल GGML_OP_CONV_2D:
            {
                // First-cut conv_2d_f32_me kernel constraints. Anything outside
                // this falls back to CPU (it's a strict subset on purpose).
                अगर (!op->src[0] || !op->src[1]) {
                    supported = false;
                    अवरोध;
                }
                अगर (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32) {
                    supported = false;
                    अवरोध;
                }
                अगर (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
                    supported = false;
                    अवरोध;
                }

                स्थिर ggml_tensor * flt = op->src[0];  // [Kw, Kh, Cin, Cout]
                स्थिर ggml_tensor * in  = op->src[1];  // [W,  H,  Cin, N]
                स्थिर int32_t       s0  = ggml_get_op_params_i32(op, 0);
                स्थिर int32_t       s1  = ggml_get_op_params_i32(op, 1);
                स्थिर int32_t       p0  = ggml_get_op_params_i32(op, 2);
                स्थिर int32_t       p1  = ggml_get_op_params_i32(op, 3);
                स्थिर int32_t       d0  = ggml_get_op_params_i32(op, 4);
                स्थिर int32_t       d1  = ggml_get_op_params_i32(op, 5);

                स्थिर int64_t Kw   = flt->ne[0];
                स्थिर int64_t Kh   = flt->ne[1];
                स्थिर int64_t Cin  = flt->ne[2];
                स्थिर int64_t Cout = flt->ne[3];
                स्थिर int64_t H    = in->ne[1];
                (व्योम) in->ne[0];

                अगर (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || in->ne[3] != 1) {
                    supported = false;
                    अवरोध;
                }
                स्थिर int64_t OW = op->ne[0];
                स्थिर int64_t OH = op->ne[1];
                अगर (OW <= 0 || OH <= 0) {
                    supported = false;
                    अवरोध;
                }
                (व्योम) p0;
                (व्योम) p1;

                // Mirror the kernel's sizing:
                //   if K_TILES * per_KT_bytes <= budget: 1 buffer, n_chunks=1
                //   else: 2 buffers (double-buffer), shrink chunk_KT until
                //         2*chunk_KT*per_KT_bytes <= budget.
                स्थिर int64_t Hp            = H + 2 * p1;
                स्थिर int64_t OW_pad        = (OW + 15) & ~15;
                स्थिर int64_t Wp_a          = OW_pad;
                स्थिर bool    need_stage    = (OW % 16 != 0);
                स्थिर int64_t stage_bytes   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                स्थिर int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                स्थिर int64_t scratch_bytes = 32 * 2 * 16 * 16 * 4;
                स्थिर int64_t budget        = L2SCP_BUDGET - stage_bytes - scratch_bytes;
                स्थिर int64_t per_KT_bytes  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                स्थिर int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                अगर (K_TILES * per_KT_bytes <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } अन्यथा {
                    chunk_KT_calc = K_TILES;
                    जबतक (chunk_KT_calc > 1 && 2 * chunk_KT_calc * per_KT_bytes > budget) {
                        chunk_KT_calc--;
                    }
                    जबतक (chunk_KT_calc > 1 && K_TILES % chunk_KT_calc != 0) {
                        chunk_KT_calc--;
                    }
                    अगर (chunk_KT_calc < 1) {
                        supported = false;
                        अवरोध;
                    }
                    n_chunks_calc = K_TILES / chunk_KT_calc;
                }

                अगर (n_chunks_calc > 1) {
                    स्थिर int64_t M_TILES     = Cout / 16;
                    स्थिर int64_t w_tiles     = (OW + 15) / 16;
                    स्थिर int64_t योग_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    स्थिर int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    अगर (योग_tiles > max_workers) {
                        supported = false;
                        अवरोध;
                    }
                }

                supported = true;
                अवरोध;
            }
        हाल GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && (ggml_nelements(op) % 16 == 0);
            अवरोध;
        हाल GGML_OP_GLU:
            // Note: we only require row-wise contiguity (ggml_is_contiguous_1) so that
            // strided views over a packed up_proj tensor (the common split-GLU layout)
            // are accepted. The kernel walks rows via nb[1] strides, so the inner
            // dimension just needs to be densely packed.
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && ggml_is_contiguous_1(op) && ggml_is_contiguous_1(op->src[0])) {
                // Check GLU variant - support SWIGLU, SWIGLU_OAI, GEGLU, GEGLU_ERF, GEGLU_QUICK, REGLU
                ggml_glu_op glu_type          = ggml_get_glu_op(op);
                स्थिर bool  supported_variant = glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_SWIGLU_OAI ||
                                                glu_type == GGML_GLU_OP_SWIGLU_CLAMP || glu_type == GGML_GLU_OP_GEGLU ||
                                                glu_type == GGML_GLU_OP_GEGLU_ERF ||
                                                glu_type == GGML_GLU_OP_GEGLU_QUICK || glu_type == GGML_GLU_OP_REGLU;

                अगर (op->src[1]) {
                    supported = supported_variant && op->src[1]->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous_1(op->src[1]) && op->src[0]->ne[0] == op->ne[0] &&
                                op->src[1]->ne[0] == op->ne[0];
                } अन्यथा {
                    supported = supported_variant && op->src[0]->ne[0] == 2 * op->ne[0];
                }
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_SOFT_MAX:
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->src[0]->ne[0] > 1) {
                // Check optional mask tensor (F32 only)
                अगर (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    अगर (!supported) {
                        अवरोध;
                    }
                }
                // Check optional sinks tensor (F32 only)
                अगर (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } अन्यथा {
                    supported = true;
                }
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_SSM_SCAN:
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] &&
                        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[2]) && op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[3]) && op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                        op->src[5] && op->src[5]->type == GGML_TYPE_F32 && op->src[6] &&
                        op->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[6]) &&
                        op->src[1]->nb[0] == माप(भग्न) && op->src[4]->nb[0] == माप(भग्न) &&
                        op->src[5]->nb[0] == माप(भग्न) &&
                        op->src[1]->nb[1] == (माप_प्रकार) op->src[1]->ne[0] * माप(भग्न) &&
                        op->src[4]->nb[1] == (माप_प्रकार) op->src[4]->ne[0] * माप(भग्न) &&
                        op->src[5]->nb[1] == (माप_प्रकार) op->src[5]->ne[0] * माप(भग्न) &&
                        op->src[0]->ne[0] == op->src[4]->ne[0] && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                        op->src[0]->ne[2] == op->src[1]->ne[1] && op->src[1]->ne[2] == op->src[2]->ne[1] &&
                        op->src[1]->ne[3] == op->src[2]->ne[2] && op->src[4]->ne[2] == op->src[1]->ne[2] &&
                        op->src[4]->ne[3] == op->src[1]->ne[3] && ggml_are_same_shape(op->src[4], op->src[5]) &&
                        op->src[6]->ne[0] == op->src[1]->ne[3] && op->src[3]->ne[1] == op->src[1]->ne[1] &&
                        (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[0]->ne[0]) &&
                        (op->src[1]->ne[1] % op->src[4]->ne[1] == 0);
            अवरोध;
        हाल GGML_OP_FLASH_ATTN_EXT:
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && op->src[2] &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == nullptr &&
                ggml_is_contiguous_rows(op) && ggml_is_contiguous_rows(op->src[0])) {
                भग्न max_bias      = 0.0f;
                भग्न logit_softcap = 0.0f;
                स_नकल(&max_bias, (स्थिर भग्न *) op->op_params + 1, माप(max_bias));
                स_नकल(&logit_softcap, (स्थिर भग्न *) op->op_params + 2, माप(logit_softcap));

                स्थिर ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                bool mask_ok = (op->src[3] == nullptr) || (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                स्थिर int64_t nhq = op->src[0]->ne[2];
                स्थिर int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                स्थिर माप_प्रकार k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                स्थिर माप_प्रकार v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                // Only support matrix engine path (F16 K/V, dk%32==0);
                // mask scalar F32 fallback to get baseline perf readings
                स्थिर bool me_eligible = op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16 &&
                                         (op->src[0]->ne[0] % 32) == 0;

                supported = me_eligible && mask_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                            max_bias == 0.0f && logit_softcap == 0.0f && op->src[0]->nb[0] == माप(भग्न) &&
                            op->src[1]->nb[0] == k_elem && op->src[2]->nb[0] == v_elem && op->nb[0] == माप(भग्न) &&
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
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_GET_ROWS:
            // Support F32/F16/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            अगर (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            अगर ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0] &&
                op->src[0]->type == op->type && ggml_is_contiguous(op)) {
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                अगर (op->तथ्य && op->src[0]->तथ्य && op->तथ्य == op->src[0]->तथ्य) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = false;
                } अन्यथा {
                    supported = true;
                }
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            अगर (op->src[0]) {
                स्थिर int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                अगर (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = true;
                } अन्यथा अगर ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    अगर (op->तथ्य && op->src[0]->तथ्य && op->तथ्य == op->src[0]->तथ्य) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = false;
                    } अन्यथा {
                        supported = true;
                    }
                } अन्यथा अगर (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = true;
                } अन्यथा {
                    supported = false;
                }
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_CONCAT:
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                स्थिर int32_t dim = ((स्थिर int32_t *) op->op_params)[0];
                अगर (dim == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Fast dim==0 path: both source row segments are cacheline-aligned
                    // and contiguous, so the kernel can use vector row copies.
                    supported = true;
                } अन्यथा अगर (dim == 0 && ((op->src[0]->nb[0] % माप(भग्न) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % माप(भग्न) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = true;
                } अन्यथा अगर (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = true;
                }
            }
            अवरोध;
        हाल GGML_OP_SSM_CONV:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == माप(भग्न) &&
                        op->src[1]->nb[0] == माप(भग्न) && op->src[0]->nb[1] == op->src[0]->ne[0] * माप(भग्न) &&
                        op->src[1]->nb[1] == op->src[1]->ne[0] * माप(भग्न) && ggml_is_contiguous(op) &&
                        op->src[1]->ne[1] == op->src[0]->ne[1] && op->ne[0] == op->src[0]->ne[1] &&
                        op->ne[1] == op->src[0]->ne[0] - op->src[1]->ne[0] + 1 && op->ne[2] == op->src[0]->ne[2];
            अवरोध;
        हाल GGML_OP_PAD:
            // F32 zero-pad only, no dim0 padding, dst contiguous
            // ne[0] must be CL-aligned (% 16 == 0) or evenly divide a CL (16 % ne[0] == 0)
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && (op->ne[0] % 16 == 0 || 16 % op->ne[0] == 0) &&
                op->src[0]->nb[0] == माप(भग्न)) {
                स्थिर int32_t lp0      = ((स्थिर int32_t *) op->op_params)[0];
                स्थिर int32_t rp0      = ((स्थिर int32_t *) op->op_params)[1];
                स्थिर bool    circular = (bool) ((स्थिर int32_t *) op->op_params)[8];
                अगर (lp0 == 0 && rp0 == 0 && !circular) {
                    supported = true;
                } अन्यथा {
                    supported = false;
                }
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_REPEAT:
            // Two acceptable shapes:
            //   1. No-op REPEAT (src and dst have identical shape): dispatched
            //      to cont_f32, which handles arbitrary contiguous sizes.
            //   2. Real REPEAT via repeat_f32 kernel: dst ne[0] cacheline-aligned,
            //      src0 ne[0] cacheline-aligned or 1, dst.ne[i] % src0.ne[i] == 0.
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op)) {
                supported = true;
            } अन्यथा अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = true;
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_FILL:
            // F32 contiguous, ne[0] cacheline-aligned for SIMD fill
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] % 16 == 0;
            अवरोध;
        हाल GGML_OP_DIAG:
            // F32 contiguous dst, src0 is 1D vector [N,1,...], dst is [N,N,...]
            // ne[0] must be cacheline-aligned for SIMD zeroing
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && op->ne[0] == op->ne[1] && op->src[0]->ne[0] == op->ne[0] &&
                        op->src[0]->ne[1] == 1 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_TRI:
            // F32 contiguous, same shape in/out
            // Kernel handles arbitrary ne[0] with aligned fast path + scalar fallback
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            अवरोध;
        हाल GGML_OP_SOLVE_TRI:
            // F32 contiguous, A square, shapes compatible
            // Only lower-triangular left-side non-unit variant
            // Require k % 16 == 0 for cache-line-safe column parallelism
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == op->src[0]->ne[1] &&
                        op->src[0]->ne[1] == op->src[1]->ne[1] && op->src[1]->ne[0] % 16 == 0 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            अवरोध;
        हाल GGML_OP_SET:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                स्थिर bool   inplace = (bool) ((स्थिर int32_t *) op->op_params)[4];
                स्थिर माप_प्रकार nb1     = ((स्थिर int32_t *) op->op_params)[0];
                स्थिर माप_प्रकार nb2     = ((स्थिर int32_t *) op->op_params)[1];
                स्थिर माप_प्रकार nb3     = ((स्थिर int32_t *) op->op_params)[2];
                स्थिर माप_प्रकार offset  = ((स्थिर int32_t *) op->op_params)[3];
                स्थिर माप_प्रकार nb0     = ggml_element_आकार(op);
                स्थिर माप_प्रकार im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                स्थिर माप_प्रकार im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                स्थिर माप_प्रकार im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                स्थिर माप_प्रकार im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                स्थिर bool view_bounds_ok = offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                स्थिर bool cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (offset % 64 == 0);

                supported = inplace && view_bounds_ok && cacheline_aligned;
            }
            अवरोध;
        हाल GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = true;
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = true;
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_GATED_DELTA_NET:
            // F32, S_v must be multiple of 8 for vectorization
            // q, k, v may be row-contiguous with strided higher dimensions.
            // g, beta, state stay contiguous.
            अगर (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&  // q
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
                supported = true;
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = true;
            अवरोध;
        हाल GGML_OP_SET_ROWS:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            अगर (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous_rows(op) &&
                ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->ne[0] == op->src[0]->ne[0]) &&              // same number of columns
                            (op->ne[2] == op->src[0]->ne[2]) &&              // same batch size
                            (op->ne[3] == op->src[0]->ne[3]) &&              // same outer dimension
                            (op->src[0]->ne[1] == op->src[1]->ne[0]) &&      // src rows = index count
                            (op->src[0]->ne[2] % op->src[1]->ne[1] == 0) &&  // batch constraint
                            (op->src[0]->ne[3] % op->src[1]->ne[2] == 0) &&  // outer constraint
                            (op->src[1]->ne[3] == 1);                        // indices tensor constraint
            } अन्यथा {
                supported = false;
            }
            अवरोध;
        हाल GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = true;
            अवरोध;
        शेष:
            supported = false;
            अवरोध;
    }
    // if(!supported) {
    //     ggml_et_dump_operator_metadata(op);
    // }
    वापस supported;
}

अटल bool ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    GGML_UNUSED(dev);
    वापस buft->iface.get_name == ggml_backend_et_बफर_type_get_name;
}

अटल bool ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    अगर (op->op == GGML_OP_GET_ROWS) {
        वापस false;
    }
    वापस true;

    GGML_UNUSED(dev);
}

अटल स्थिर काष्ठा ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ शून्य,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_et_graph_compute,
    /* .event_record            = */ शून्य,
    /* .event_wait              = */ शून्य,
    /* .graph_optimize          = */ शून्य,
};

अटल स्थिर अक्षर * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    वापस dev_ctx->name.c_str();
}

अटल स्थिर अक्षर * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    वापस dev_ctx->desc.c_str();
}

अटल व्योम ggml_backend_et_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *मुक्त                                    = dev_ctx->योग_mem;
    *योग                                   = dev_ctx->योग_mem;
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    वापस GGML_BACKEND_DEVICE_TYPE_GPU;
}

अटल व्योम ggml_backend_et_device_get_props(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props) {
    GGML_UNUSED(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->device_id = शून्य;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

अटल ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    GGML_UNUSED(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    वापस ggml_backend_et_init(dev_ctx->devidx);
}

अटल ggml_backend_बफर_type_t ggml_backend_et_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    वापस dev_ctx->buftype;
}

अटल ggml_backend_बफर_type_t ggml_backend_et_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    वापस ggml_backend_cpu_बफर_type();
}

अटल स्थिर काष्ठा ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_memory,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_बफर_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_बफर_type,
    /* .buffer_from_host_ptr = */ शून्य,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ शून्य,
    /* .event_free        = */ शून्य,
    /* .event_synchronize = */ शून्य,
};

/*
  Backend Registry.
*/

अटल स्थिर अक्षर * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    वापस GGML_ET_NAME;
}

अटल माप_प्रकार ggml_backend_et_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    वापस ctx->devices.size();
}

अटल ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    अगर (devidx >= ctx->devices.size()) {
        वापस nullptr;
    }
    वापस ctx->devices[devidx];
}

अटल व्योम * ggml_backend_et_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    वापस nullptr;
}

अटल स्थिर काष्ठा ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_address,
};

ggml_backend_reg_t ggml_backend_et_reg(व्योम) {
    अटल ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = नया ggml_backend_et_reg_ctx;

        अगर (!ggml_et_driver_init()) {
            वापस nullptr;
        }

        ggml_backend_reg_t r = नया ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ nullptr,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        क्रम (पूर्णांक i = 0; i < ggml_et_devicelayer()->get_Devices_गिनती(); i++) {
            ggml_backend_dev_t dev = नया ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ nullptr  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = नया ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->योग_mem                       = static_cast<माप_प्रकार>(prop.memory_आकार);
            {
                स्थिर अक्षर * env            = दो_पर्या("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && म_भेद(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_बफर_type_context * bufty_ctx = नया ggml_backend_et_बफर_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = नया ggml_backend_बफर_type{ /* .iface   = */ ggml_backend_et_बफर_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_stream = ggml_et_runtime()->createStream(rtid);

            dev_ctx->trace_बफर = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_बफर_आकार);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            क्रम (स्वतः & slot : dev_ctx->uberkernel.slots) {
                slot.insts.reserve(256);
                slot.params_blob.reserve(1 << 20);
                slot.device_insts_capacity  = 256 * माप(ggml_et_uberkernel_inst);
                slot.device_params_capacity = 1 << 20;
                slot.device_insts           = ggml_et_runtime()->mallocDevice(rtid, slot.device_insts_capacity);
                slot.device_params          = ggml_et_runtime()->mallocDevice(rtid, slot.device_params_capacity);
                अगर (slot.device_insts == nullptr) {
                    slot.device_insts_capacity = 0;
                }
                अगर (slot.device_params == nullptr) {
                    slot.device_params_capacity = 0;
                }
            }

            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
        }

        r->context = ctx;
        वापस r;
    }();

    वापस _reg;
}

ggml_guid_t ggml_backend_et_guid(व्योम) {
    अटल ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    वापस &guid;
}

ggml_backend_t ggml_backend_et_init(माप_प्रकार devidx) {
    अगर (!ggml_et_driver_init()) {
        वापस nullptr;
    }

    अगर (devidx >= (माप_प्रकार) ggml_backend_et_get_device_गिनती()) {
        वापस nullptr;
    }

    ggml_backend_et_context * ctx = नया ggml_backend_et_context;
    ctx->devidx                   = (पूर्णांक) devidx;

    ggml_backend_t backend = नया ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    वापस backend;
}

bool ggml_backend_is_et(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

पूर्णांक ggml_backend_et_get_device_गिनती(व्योम) {
    वापस ggml_backend_et_reg_get_device_गिनती(ggml_backend_et_reg());
}

व्योम ggml_backend_et_get_device_description(पूर्णांक devidx, अक्षर * description, माप_प्रकार description_आकार) {
    अगर (devidx < 0 || devidx >= ggml_backend_et_get_device_गिनती()) {
        snprintf(description, description_आकार, "ET Device %d (invalid)", devidx);
        वापस;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_आकार, "%s", dev_ctx->desc.c_str());
}

व्योम ggml_backend_et_get_device_memory(पूर्णांक devidx, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    अगर (devidx < 0 || devidx >= ggml_backend_et_get_device_गिनती()) {
        *मुक्त  = 0;
        *योग = 0;
        वापस;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_memory(dev, मुक्त, योग);
}

ggml_backend_बफर_type_t ggml_backend_et_बफर_type(माप_प्रकार dev_अंक) {
    अगर (dev_अंक >= (माप_प्रकार) ggml_backend_et_get_device_गिनती()) {
        वापस nullptr;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_अंक);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    वापस dev_ctx->buftype;
}

ggml_backend_बफर_type_t ggml_backend_et_host_बफर_type(व्योम) {
    अटल ggml_backend_बफर_type host_बफर_type = {
        /* .iface   = */ ggml_backend_et_बफर_type_i,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    वापस &host_बफर_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)

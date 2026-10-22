#অন্তর্ভাব "ggml-et.h"

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-et-common.h"
#অন্তর্ভাব "ggml-et-kernels.h"
#অন্তর্ভাব "ggml-et-memops.h"
#অন্তর্ভাব "ggml-et-ops.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <stdarg.h>

#অন্তর্ভাব <cstdarg>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <string>
#অন্তর্ভাব <vector>

#যদি __has_include(<filesystem>)
#    include <filesystem>
নামাকাশ fs = std::filesystem;
#নতুবা_যদি __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
নামাকাশ fs = std::experimental::filesystem;
#অন্যথা
#    error "cannot include the filesystem library"
#endif

/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
স্থির শূন্য ggml_et_dump_tensor_metadata(ধ্রুবক ggml_tensor * ggtensor, size_t indent_level, ধ্রুবক অক্ষর * title) {
    std::string spaces(indent_level, ' ');
    fprintf(stderr,
            "%s%s: %s\n"
            "%s  type: %s\n"
            "%s  ne: %lld %lld %lld %lld\n"
            "%s  nb: %zu %zu %zu %zu\n"
            "%s  op: %s\n"
            "%s  data: %p\n"
            "%s  src0: %p\n",
            spaces.c_str(), title, ggtensor->name, spaces.c_str(), ggml_type_name(ggtensor->type), spaces.c_str(), (long long) ggtensor->ne[0],
            (long long) ggtensor->ne[1], (long long) ggtensor->ne[2], (long long) ggtensor->ne[3], spaces.c_str(),
            ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3], spaces.c_str(), ggml_op_name(ggtensor->op),
            spaces.c_str(), ggtensor->তথ্য, spaces.c_str(), (শূন্য *) ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
স্থির শূন্য ggml_et_dump_operator_metadata(ধ্রুবক ggml_tensor * ggtensor) {
    GGML_ASSERT(ggtensor != ফাঁকা);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_SRC && ggtensor->src[i]; i++) {
        অক্ষর arr[16];
        পূর্ণসংখ্যা  n = snprintf(arr, আকার(arr), "src[%i]->name", i);
        GGML_ASSERT((unsigned) n < আকার(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    যদি (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

স্থির কাঠামো ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::ofstream>                    profile_stream;
    std::unique_ptr<std::ofstream>                    kernel_id_stream;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    যৌক্তিক                                              profiling_enabled = মিথ্যা;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
স্থির std::string ggml_et_get_default_et_path() {
    // List of environment variables to check in order of preference
    ধ্রুবক অক্ষর * ধ্রুবক env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    জন্য (ধ্রুবক অক্ষর * var : env_vars) {
        যদি (ধ্রুবক অক্ষর * et_path = std::getenv(var)) {
            যদি (et_path && *et_path != '\0') {
                ফেরত fs::path(et_path).string();
            }
        }
    }

    // Otherwise assume default
    ফেরত fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
স্থির অন্তঃরেখ auto ggml_et_get_default_sysemu_options() {
    constexpr uint64_t kSysEmuMaxCycles        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t kSysEmuMinionShiresMask = 0x1FFFFFFFFu;
    ধ্রুবক std::string  et_path                 = ggml_et_get_default_et_path() + "/";

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
    ধ্রুবক std::vector<std::string> required_files = {
        sysEmuOptions.bootromTrampolineToBL2ElfPath, sysEmuOptions.spBL2ElfPath,
        sysEmuOptions.machineMinionElfPath,          sysEmuOptions.masterMinionElfPath,
        sysEmuOptions.workerMinionElfPath,           sysEmuOptions.executablePath,
    };

    জন্য (ধ্রুবক auto & file : required_files) {
        যদি (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.runDir           = (fs::current_path().string() + "/");
    sysEmuOptions.maxCycles        = kSysEmuMaxCycles;
    sysEmuOptions.minionShiresMask = kSysEmuMinionShiresMask;
    sysEmuOptions.puUart0Path      = sysEmuOptions.runDir + "pu_uart0_tx.log";
    sysEmuOptions.puUart1Path      = sysEmuOptions.runDir + "pu_uart1_tx.log";
    sysEmuOptions.spUart0Path      = sysEmuOptions.runDir + "spio_uart0_tx.log";
    sysEmuOptions.spUart1Path      = sysEmuOptions.runDir + "spio_uart1_tx.log";
    sysEmuOptions.startGdb         = মিথ্যা;
    sysEmuOptions.memcheck         = মিথ্যা;

    ফেরত sysEmuOptions;
}

// Forward declaration
স্থির শূন্য ggml_et_driver_cleanup();

স্থির যৌক্তিক ggml_et_driver_init() {
    যদি (_drv.runtime != nullptr) {
        প্রতিজ্ঞা(_drv.device_layer != nullptr);
    } অন্যথা {
        চেষ্টা {
#যদি defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#অন্যথা
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#endif  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            ধ্রুবক অক্ষর * profile_path = getenv("GGML_ET_PROFILE");
            যদি (profile_path) {
                std::string output_path    = std::string(profile_path) + "/et_runtime_trace.json";
                std::string kernel_id_path = std::string(profile_path) + "/kernel_id.json";

                _drv.profile_stream   = std::make_unique<std::ofstream>(output_path);
                _drv.kernel_id_stream = std::make_unique<std::ofstream>(kernel_id_path);
                যদি (!_drv.profile_stream->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_path.c_str());
                    abort();
                }
                যদি (!_drv.kernel_id_stream->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_path.c_str());
                    abort();
                }

                auto * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_stream, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = সত্য;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::atexit(ggml_et_driver_cleanup);
            }
        } ধরো (ধ্রুবক std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            যদি (_drv.device_layer != nullptr) {
                _drv.device_layer.reset();
            }
            যদি (_drv.runtime != nullptr) {
                _drv.runtime.reset();
            }
            ফেরত মিথ্যা;
        }
    }
    ফেরত সত্য;
}

স্থির std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    ফেরত _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    ফেরত _drv.runtime;
}

স্থির শূন্য ggml_et_driver_cleanup() {
    যদি (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        auto * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = মিথ্যা;

        যদি (_drv.profile_stream) {
            _drv.profile_stream->close();
            _drv.profile_stream.reset();
        }

        // Save kernel map
        যদি (_drv.kernel_id_stream && !_drv.kernel_map.empty()) {
            auto & os = *_drv.kernel_id_stream;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            জন্য (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].first << "\": " << (পূর্ণসংখ্যা) _drv.kernel_map[i].second;
                যদি (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_stream->close();
            _drv.kernel_id_stream.reset();
        }
    }
}

স্থির ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

স্থির শূন্য ggml_backend_et_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    ggml_backend_et_বাফার_context * ctx = (ggml_backend_et_বাফার_context *) বাফার->context;
    যদি (ctx->তথ্য != nullptr) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        যদি (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->তথ্য));
        }
    }
    delete ctx;
}

স্থির শূন্য * ggml_backend_et_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    ggml_backend_et_বাফার_context * ctx = (ggml_backend_et_বাফার_context *) বাফার->context;
    ফেরত ctx->তথ্য;
}

স্থির ggml_status ggml_backend_et_বাফার_init_tensor(ggml_backend_বাফার_t বাফার, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    যদি (tensor->view_src != ফাঁকা) {
        ফেরত GGML_STATUS_SUCCESS;
    }

    ধ্রুবক size_t original_আকার = ggml_nbytes(tensor);
    ধ্রুবক size_t padded_আকার   = ggml_backend_buft_get_alloc_আকার(বাফার->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    যদি (padded_আকার > original_আকার) {
        ধ্রুবক size_t padding_আকার = padded_আকার - original_আকার;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) বাফার->buft->device->context;
        যদি (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            ফেরত GGML_STATUS_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->তথ্য) + original_আকার;
        যদি (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_আকার)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            ফেরত GGML_STATUS_FAILED;
        }
    }

    ফেরত GGML_STATUS_SUCCESS;
}

স্থির শূন্য ggml_backend_et_বাফার_set_tensor(ggml_backend_বাফার_t বাফার,
                                              ggml_tensor *         tensor,
                                              ধ্রুবক শূন্য *          তথ্য,
                                              size_t                offset,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        ফেরত;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) বাফার->buft->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->তথ্য) + offset;
    ধ্রুবক std::byte * src_ptr = static_cast<ধ্রুবক std::byte *>(তথ্য);

    rt::EventId event = runtime->memcpyHostToDevice(stream, src_ptr, dst_ptr, size, সত্য /*barrier*/);

    runtime->waitForEvent(event);
}

স্থির শূন্য ggml_backend_et_বাফার_get_tensor(ggml_backend_বাফার_t বাফার,
                                              ধ্রুবক ggml_tensor *   tensor,
                                              শূন্য *                তথ্য,
                                              size_t                offset,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        ফেরত;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) বাফার->buft->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    ধ্রুবক std::byte * src_ptr = static_cast<ধ্রুবক std::byte *>(tensor->তথ্য) + offset;
    std::byte *       dst_ptr = static_cast<std::byte *>(তথ্য);

    rt::EventId event = runtime->memcpyDeviceToHost(stream, src_ptr, dst_ptr, size, সত্য /*barrier*/);

    runtime->waitForEvent(event);
}

স্থির যৌক্তিক ggml_backend_et_বাফার_cpy_tensor(ggml_backend_বাফার_t বাফার,
                                              ধ্রুবক ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_UNUSED(বাফার);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    ফেরত মিথ্যা;
}

স্থির শূন্য ggml_backend_et_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    ggml_backend_et_বাফার_context * ctx = (ggml_backend_et_বাফার_context *) বাফার->context;

    যদি (ctx->size == 0 || ctx->তথ্য == nullptr) {
        ফেরত;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) বাফার->buft->device->context;
    যদি (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        ফেরত;
    }

    // Use device-side memset kernel for efficient clearing
    যদি (!ggml_et_memset(dev_ctx, ctx->তথ্য, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        ফেরত;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

স্থির ধ্রুবক কাঠামো ggml_backend_বাফার_i ggml_backend_et_বাফার_i = {
    /* .free_buffer     = */ ggml_backend_et_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_et_বাফার_get_base,
    /* .init_tensor     = */ ggml_backend_et_বাফার_init_tensor,
    /* .memset_tensor   = */ ফাঁকা,
    /* .set_tensor      = */ ggml_backend_et_বাফার_set_tensor,
    /* .get_tensor      = */ ggml_backend_et_বাফার_get_tensor,
    /* .set_tensor_2d   = */ ফাঁকা,
    /* .get_tensor_2d   = */ ফাঁকা,
    /* .cpy_tensor      = */ ggml_backend_et_বাফার_cpy_tensor,
    /* .clear           = */ ggml_backend_et_বাফার_clear,
    /* .reset           = */ ফাঁকা,
};

স্থির ধ্রুবক অক্ষর * ggml_backend_et_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    GGML_UNUSED(buft);
    ফেরত GGML_ET_NAME;
}

স্থির ggml_backend_বাফার_t ggml_backend_et_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft, size_t size) {
    ggml_backend_et_বাফার_type_context * btctx = (ggml_backend_et_বাফার_type_context *) buft->context;

    ggml_backend_et_বাফার_context * ctx = নতুন ggml_backend_et_বাফার_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        delete ctx;
        ফেরত nullptr;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    যদি (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        delete ctx;
        ফেরত nullptr;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->তথ্য = runtime->mallocDevice(ctx->rtid, size);
    যদি (ctx->তথ্য == nullptr) {
        delete ctx;
        ফেরত nullptr;
    }

    ফেরত ggml_backend_বাফার_init(buft, ggml_backend_et_বাফার_i, ctx, size);
}

স্থির size_t ggml_backend_et_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime || !buft->device) {
        ফেরত GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    ফেরত prop.cache_সারি_আকার;
}

স্থির size_t ggml_backend_et_বাফার_type_get_max_আকার(ggml_backend_বাফার_type_t buft) {
    যদি (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        ফেরত dev_ctx->যোগ_mem;
    }
    ফেরত SIZE_MAX;
}

স্থির size_t ggml_backend_et_বাফার_type_get_alloc_আকার(ggml_backend_বাফার_type_t buft, ধ্রুবক ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    ফেরত ggml_nbytes_pad(tensor);
}

স্থির যৌক্তিক ggml_backend_et_বাফার_type_is_host(ggml_backend_বাফার_type_t buft) {
    GGML_UNUSED(buft);
    ফেরত মিথ্যা;
}

স্থির ধ্রুবক কাঠামো ggml_backend_বাফার_type_i ggml_backend_et_বাফার_type_i = {
    /* .get_name            = */ ggml_backend_et_বাফার_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_বাফার_type_alloc_বাফার,
    /* .alloc_buffer_n      = */ ফাঁকা,
    /* .get_alignment       = */ ggml_backend_et_বাফার_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_বাফার_type_get_max_আকার,
    /* .get_alloc_size      = */ ggml_backend_et_বাফার_type_get_alloc_আকার,
    /* .get_alloc_size_n    = */ ফাঁকা,
    /* .is_host             = */ ggml_backend_et_বাফার_type_is_host,
};

স্থির ধ্রুবক অক্ষর * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    ফেরত GGML_ET_NAME;
}

স্থির শূন্য ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context *     et_ctx  = (ggml_backend_et_context *) backend->context;
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    যদি (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;

        যদি (_drv.profiling_enabled) {
            auto kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        যদি (runtime) {
            যদি (dev_ctx->trace_বাফার) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_বাফার);
                dev_ctx->trace_বাফার = nullptr;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->waitForStream(dev_ctx->default_stream);
            জন্য (auto & slot : dev_ctx->uberkernel.slots) {
                যদি (slot.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
                    slot.device_insts = nullptr;
                }
                যদি (slot.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, slot.device_params);
                    slot.device_params = nullptr;
                }
                slot.has_pending = মিথ্যা;
            }
        }
    }

    delete et_ctx;
    delete backend;
}

স্থির ggml_backend_বাফার_type_t ggml_backend_et_get_default_বাফার_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    ফেরত ggml_backend_et_বাফার_type(et_ctx->devidx);
}

স্থির শূন্য ggml_backend_et_set_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             ধ্রুবক শূন্য *   তথ্য,
                                             size_t         offset,
                                             size_t         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        ফেরত;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->তথ্য) + offset;
    ধ্রুবক std::byte * src_ptr = static_cast<ধ্রুবক std::byte *>(তথ্য);

    runtime->memcpyHostToDevice(stream, src_ptr, dst_ptr, size, সত্য /*barrier*/);
}

স্থির শূন্য ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             ধ্রুবক ggml_tensor * tensor,
                                             শূন্য *              তথ্য,
                                             size_t              offset,
                                             size_t              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        ফেরত;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::StreamId                     stream  = dev_ctx->default_stream;

    ধ্রুবক std::byte * src_ptr = static_cast<ধ্রুবক std::byte *>(tensor->তথ্য) + offset;
    std::byte *       dst_ptr = static_cast<std::byte *>(তথ্য);

    runtime->memcpyDeviceToHost(stream, src_ptr, dst_ptr, size, সত্য /*barrier*/);
}

স্থির যৌক্তিক ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             ধ্রুবক ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    ফেরত মিথ্যা;
}

স্থির শূন্য ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        ফেরত;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->waitForStream(dev_ctx->default_stream);

    auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
    যদি (errors.empty()) {
        ফেরত;
    }
    জন্য (ধ্রুবক auto & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (পূর্ণসংখ্যা) err.errorCode_,
                       (পূর্ণসংখ্যা) err.errorContext_.value()[0].type_);
    }
    abort();
}

স্থির যৌক্তিক ggml_et_can_fuse(ধ্রুবক ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx, std::initializer_list<ggml_op> ops) {
    যদি (!ggml_can_fuse(cgraph, গ্রন্থি_idx, ops)) {
        ফেরত মিথ্যা;
    }

    যদি (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        ধ্রুবক ggml_tensor * mm  = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * add = cgraph->nodes[গ্রন্থি_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        যদি (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        যদি (add->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }
        যদি (add->src[0] != mm && add->src[1] != mm) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক ggml_tensor * bias = (add->src[0] == mm) ? add->src[1] : add->src[0];

        যদি (bias->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }

        // No broadcasting: bias shape must equal MM output shape.
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; ++i) {
            যদি (bias->ne[i] != mm->ne[i]) {
                ফেরত মিথ্যা;
            }
        }

        // Bias and dst must be contiguous and have identical strides - the
        // kernel uses dst-style offset arithmetic against bias's nb[].
        যদি (!ggml_is_contiguous(bias) || !ggml_is_contiguous(mm)) {
            ফেরত মিথ্যা;
        }
        জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; ++i) {
            যদি ((int64_t) bias->nb[i] != (int64_t) add->nb[i]) {
                ফেরত মিথ্যা;
            }
        }
    }

    যদি (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ধ্রুবক ggml_tensor * rms_norm = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * mul      = cgraph->nodes[গ্রন্থি_idx + 1];

        // ET only supports F32
        যদি (rms_norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        ধ্রুবক ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        যদি (weights->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        যদি (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_rows(weights)) {
            ফেরত মিথ্যা;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        যদি (rms_norm->src[0]->ne[0] % 16 != 0 || weights->ne[0] % 16 != 0) {
            ফেরত মিথ্যা;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        যদি (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

স্থির ggml_status ggml_backend_et_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_graph(&dev_ctx->uberkernel);

    জন্য (পূর্ণসংখ্যা i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * গ্রন্থি = cgraph->nodes[i];

        যদি (গ্রন্থি->op == GGML_OP_NONE || গ্রন্থি->op == GGML_OP_VIEW || গ্রন্থি->op == GGML_OP_RESHAPE ||
            গ্রন্থি->op == GGML_OP_PERMUTE || গ্রন্থি->op == GGML_OP_TRANSPOSE) {
            চালু;
        }

        // --- Fusion checks (before regular dispatch) ---
        যদি (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, গ্রন্থি, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            চালু;
        }
        যদি (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
            ggml_et_op_mul_mat(dev_ctx, গ্রন্থি, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            চালু;
        }

        নির্বাচন (গ্রন্থি->op) {
            ক্ষেত্র GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_যোগ_ROWS:
                ggml_et_op_যোগ_rows(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_MEAN:
                ggml_et_op_mean(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_ADD:
                ggml_et_op_add(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_GROUP_NORM:
                ggml_et_op_group_norm(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_IM2COL:
                ggml_et_op_im2col(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_GET_ROWS:
                ggml_et_op_get_rows(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_SSM_SCAN:
                ggml_et_op_ssm_scan(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_SET_ROWS:
                ggml_et_op_set_rows(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_SET:
                ggml_et_op_set(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, গ্রন্থি);
                বিরতি;

            ক্ষেত্র GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, গ্রন্থি);
                বিরতি;

            পূর্বনির্ধারিত:
                ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(গ্রন্থি->op));
                ফেরত GGML_STATUS_FAILED;
        }

        যদি (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
            ফেরত GGML_STATUS_FAILED;
        }
    }

    যদি (!ggml_et_uberkernel_end_graph(dev_ctx)) {
        ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
        ফেরত GGML_STATUS_FAILED;
    }

    ফেরত GGML_STATUS_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
স্থির যৌক্তিক et_ggml_is_row_contiguous(ধ্রুবক ggml_tensor * t) {
    ফেরত t->nb[0] == ggml_type_আকার(t->type);
}

স্থির যৌক্তিক ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, ধ্রুবক ggml_tensor * op) {
    GGML_UNUSED(dev);

    যৌক্তিক supported = মিথ্যা;
    নির্বাচন (op->op) {
        ক্ষেত্র GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == আকার(দশমিক) && ggml_is_contiguous(op);
            বিরতি;
        ক্ষেত্র GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_যোগ_ROWS:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_MEAN:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == আকার(দশমিক) && op->src[0]->nb[0] == আকার(দশমিক)) {
                নির্বাচন (ggml_get_unary_op(op)) {
                    ক্ষেত্র GGML_UNARY_OP_ABS:
                    ক্ষেত্র GGML_UNARY_OP_SGN:
                    ক্ষেত্র GGML_UNARY_OP_NEG:
                    ক্ষেত্র GGML_UNARY_OP_STEP:
                    ক্ষেত্র GGML_UNARY_OP_TANH:
                    ক্ষেত্র GGML_UNARY_OP_ELU:
                    ক্ষেত্র GGML_UNARY_OP_RELU:
                    ক্ষেত্র GGML_UNARY_OP_SIGMOID:
                    ক্ষেত্র GGML_UNARY_OP_GELU:
                    ক্ষেত্র GGML_UNARY_OP_GELU_QUICK:
                    ক্ষেত্র GGML_UNARY_OP_SILU:
                    ক্ষেত্র GGML_UNARY_OP_HARDSWISH:
                    ক্ষেত্র GGML_UNARY_OP_HARDSIGMOID:
                    ক্ষেত্র GGML_UNARY_OP_EXP:
                    ক্ষেত্র GGML_UNARY_OP_EXPM1:
                    ক্ষেত্র GGML_UNARY_OP_SOFTPLUS:
                    ক্ষেত্র GGML_UNARY_OP_GELU_ERF:
                    ক্ষেত্র GGML_UNARY_OP_FLOOR:
                    ক্ষেত্র GGML_UNARY_OP_CEIL:
                    ক্ষেত্র GGML_UNARY_OP_ROUND:
                    ক্ষেত্র GGML_UNARY_OP_TRUNC:
                        supported = সত্য;
                        বিরতি;
                    পূর্বনির্ধারিত:
                        বিরতি;
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_MUL:
        ক্ষেত্র GGML_OP_ADD:
        ক্ষেত্র GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->nb[0] == আকার(দশমিক) &&
                        op->src[0]->nb[0] == আকার(দশমিক) &&
                        (op->src[1]->nb[0] == আকার(দশমিক) || op->src[1]->ne[0] == 1) &&
                        op->nb[1] == op->ne[0] * আকার(দশমিক);
            বিরতি;
        ক্ষেত্র GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            যদি (op->type == GGML_TYPE_F32 &&
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
                supported = সত্য;
            } অন্যথা যদি (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                যৌক্তিক src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_আকার(op->src[0]->type));
                যৌক্তিক src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_আকার(op->src[1]->type));
                যৌক্তিক dst_first_dim_contiguous  = (op->nb[0] == আকার(দশমিক));

                // Check destination stride ordering (only for dimensions with ne > 1)
                যৌক্তিক dst_properly_ordered = সত্য;
                জন্য (পূর্ণসংখ্যা d = 0; d < 3; d++) {
                    যদি (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = মিথ্যা;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;
            } অন্যথা যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                যৌক্তিক src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_আকার(op->src[0]->type));
                যৌক্তিক src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_আকার(op->src[1]->type));
                যৌক্তিক dst_first_dim_contiguous  = (op->nb[0] == আকার(দশমিক));

                যৌক্তিক dst_properly_ordered = সত্য;
                জন্য (পূর্ণসংখ্যা d = 0; d < 3; d++) {
                    যদি (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = মিথ্যা;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;

            } অন্যথা যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                যৌক্তিক src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_আকার(op->src[0]->type));
                যৌক্তিক src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_আকার(op->src[1]->type));
                যৌক্তিক dst_first_dim_contiguous  = (op->nb[0] == আকার(দশমিক));

                যৌক্তিক dst_properly_ordered = সত্য;
                জন্য (পূর্ণসংখ্যা d = 0; d < 3; d++) {
                    যদি (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = মিথ্যা;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/Q4_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            যদি (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_I32) {
                // Check first dimension contiguity requirements (matching CPU backend)
                যৌক্তিক src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_আকার(op->src[0]->type));
                যৌক্তিক src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_আকার(op->src[1]->type));
                যৌক্তিক src2_first_dim_contiguous = (op->src[2]->nb[0] == ggml_type_আকার(op->src[2]->type));
                যৌক্তিক dst_first_dim_contiguous  = (op->nb[0] == আকার(দশমিক));

                // Check destination stride ordering (only for dimensions with ne > 1)
                যৌক্তিক dst_properly_ordered = সত্য;
                জন্য (পূর্ণসংখ্যা d = 0; d < 3; d++) {
                    যদি (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = মিথ্যা;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                যৌক্তিক dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                  (op->src[1]->ne[3] == 1) &&  // b is 3d
                                  (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                  (op->src[2]->ne[1] == op->src[1]->ne[2]) &&    // must have expert list per b row
                                  (op->src[0]->ne[0] == op->src[1]->ne[0]) &&    // K dimension must match
                                  (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && src2_first_dim_contiguous &&
                            dst_first_dim_contiguous && dst_properly_ordered && dims_valid;
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0])) {
                ধ্রুবক পূর্ণসংখ্যা  mode             = ggml_get_op_params_i32(op, 2);
                ধ্রুবক পূর্ণসংখ্যা  ndims            = ggml_get_op_params_i32(op, 1);
                ধ্রুবক যৌক্তিক is_normal        = mode == GGML_ROPE_TYPE_NORMAL;
                ধ্রুবক যৌক্তিক is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                ধ্রুবক যৌক্তিক is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                ধ্রুবক যৌক্তিক zero_view_offset = op->src[0]->view_src == nullptr || op->src[0]->view_offs == 0;
                ধ্রুবক যৌক্তিক has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                ধ্রুবক যৌক্তিক zero_rot_offset  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_offset && zero_rot_offset && ndims <= 512 &&
                    (is_normal || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_GROUP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]) &&
                        ggml_get_op_params_i32(op, 0) > 0;
            বিরতি;
        ক্ষেত্র GGML_OP_IM2COL:
            supported = op->src[0] && op->src[1] &&
                        ((op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                         (op->type == GGML_TYPE_F16 &&
                          (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32))) &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) &&
                        op->nb[0] == ggml_type_আকার(op->type) && op->src[1]->nb[0] == ggml_type_আকার(op->src[1]->type);
            বিরতি;
        ক্ষেত্র GGML_OP_CONV_2D:
            {
                // First-cut conv_2d_f32_me kernel constraints. Anything outside
                // this falls back to CPU (it's a strict subset on purpose).
                যদি (!op->src[0] || !op->src[1]) {
                    supported = মিথ্যা;
                    বিরতি;
                }
                যদি (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32) {
                    supported = মিথ্যা;
                    বিরতি;
                }
                যদি (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
                    supported = মিথ্যা;
                    বিরতি;
                }

                ধ্রুবক ggml_tensor * flt = op->src[0];  // [Kw, Kh, Cin, Cout]
                ধ্রুবক ggml_tensor * in  = op->src[1];  // [W,  H,  Cin, N]
                ধ্রুবক int32_t       s0  = ggml_get_op_params_i32(op, 0);
                ধ্রুবক int32_t       s1  = ggml_get_op_params_i32(op, 1);
                ধ্রুবক int32_t       p0  = ggml_get_op_params_i32(op, 2);
                ধ্রুবক int32_t       p1  = ggml_get_op_params_i32(op, 3);
                ধ্রুবক int32_t       d0  = ggml_get_op_params_i32(op, 4);
                ধ্রুবক int32_t       d1  = ggml_get_op_params_i32(op, 5);

                ধ্রুবক int64_t Kw   = flt->ne[0];
                ধ্রুবক int64_t Kh   = flt->ne[1];
                ধ্রুবক int64_t Cin  = flt->ne[2];
                ধ্রুবক int64_t Cout = flt->ne[3];
                ধ্রুবক int64_t H    = in->ne[1];
                (শূন্য) in->ne[0];

                যদি (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || in->ne[3] != 1) {
                    supported = মিথ্যা;
                    বিরতি;
                }
                ধ্রুবক int64_t OW = op->ne[0];
                ধ্রুবক int64_t OH = op->ne[1];
                যদি (OW <= 0 || OH <= 0) {
                    supported = মিথ্যা;
                    বিরতি;
                }
                (শূন্য) p0;
                (শূন্য) p1;

                // Mirror the kernel's sizing:
                //   if K_TILES * per_KT_bytes <= budget: 1 buffer, n_chunks=1
                //   else: 2 buffers (double-buffer), shrink chunk_KT until
                //         2*chunk_KT*per_KT_bytes <= budget.
                ধ্রুবক int64_t Hp            = H + 2 * p1;
                ধ্রুবক int64_t OW_pad        = (OW + 15) & ~15;
                ধ্রুবক int64_t Wp_a          = OW_pad;
                ধ্রুবক যৌক্তিক    need_stage    = (OW % 16 != 0);
                ধ্রুবক int64_t stage_bytes   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                ধ্রুবক int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                ধ্রুবক int64_t scratch_bytes = 32 * 2 * 16 * 16 * 4;
                ধ্রুবক int64_t budget        = L2SCP_BUDGET - stage_bytes - scratch_bytes;
                ধ্রুবক int64_t per_KT_bytes  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                ধ্রুবক int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                যদি (K_TILES * per_KT_bytes <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } অন্যথা {
                    chunk_KT_calc = K_TILES;
                    যখন (chunk_KT_calc > 1 && 2 * chunk_KT_calc * per_KT_bytes > budget) {
                        chunk_KT_calc--;
                    }
                    যখন (chunk_KT_calc > 1 && K_TILES % chunk_KT_calc != 0) {
                        chunk_KT_calc--;
                    }
                    যদি (chunk_KT_calc < 1) {
                        supported = মিথ্যা;
                        বিরতি;
                    }
                    n_chunks_calc = K_TILES / chunk_KT_calc;
                }

                যদি (n_chunks_calc > 1) {
                    ধ্রুবক int64_t M_TILES     = Cout / 16;
                    ধ্রুবক int64_t w_tiles     = (OW + 15) / 16;
                    ধ্রুবক int64_t যোগ_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    ধ্রুবক int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    যদি (যোগ_tiles > max_workers) {
                        supported = মিথ্যা;
                        বিরতি;
                    }
                }

                supported = সত্য;
                বিরতি;
            }
        ক্ষেত্র GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && (ggml_nelements(op) % 16 == 0);
            বিরতি;
        ক্ষেত্র GGML_OP_GLU:
            // Note: we only require row-wise contiguity (ggml_is_contiguous_1) so that
            // strided views over a packed up_proj tensor (the common split-GLU layout)
            // are accepted. The kernel walks rows via nb[1] strides, so the inner
            // dimension just needs to be densely packed.
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && ggml_is_contiguous_1(op) && ggml_is_contiguous_1(op->src[0])) {
                // Check GLU variant - support SWIGLU, SWIGLU_OAI, GEGLU, GEGLU_ERF, GEGLU_QUICK, REGLU
                ggml_glu_op glu_type          = ggml_get_glu_op(op);
                ধ্রুবক যৌক্তিক  supported_variant = glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_SWIGLU_OAI ||
                                                glu_type == GGML_GLU_OP_SWIGLU_CLAMP || glu_type == GGML_GLU_OP_GEGLU ||
                                                glu_type == GGML_GLU_OP_GEGLU_ERF ||
                                                glu_type == GGML_GLU_OP_GEGLU_QUICK || glu_type == GGML_GLU_OP_REGLU;

                যদি (op->src[1]) {
                    supported = supported_variant && op->src[1]->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous_1(op->src[1]) && op->src[0]->ne[0] == op->ne[0] &&
                                op->src[1]->ne[0] == op->ne[0];
                } অন্যথা {
                    supported = supported_variant && op->src[0]->ne[0] == 2 * op->ne[0];
                }
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_SOFT_MAX:
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->src[0]->ne[0] > 1) {
                // Check optional mask tensor (F32 only)
                যদি (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    যদি (!supported) {
                        বিরতি;
                    }
                }
                // Check optional sinks tensor (F32 only)
                যদি (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } অন্যথা {
                    supported = সত্য;
                }
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_SSM_SCAN:
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] &&
                        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[2]) && op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[3]) && op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                        op->src[5] && op->src[5]->type == GGML_TYPE_F32 && op->src[6] &&
                        op->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[6]) &&
                        op->src[1]->nb[0] == আকার(দশমিক) && op->src[4]->nb[0] == আকার(দশমিক) &&
                        op->src[5]->nb[0] == আকার(দশমিক) &&
                        op->src[1]->nb[1] == (size_t) op->src[1]->ne[0] * আকার(দশমিক) &&
                        op->src[4]->nb[1] == (size_t) op->src[4]->ne[0] * আকার(দশমিক) &&
                        op->src[5]->nb[1] == (size_t) op->src[5]->ne[0] * আকার(দশমিক) &&
                        op->src[0]->ne[0] == op->src[4]->ne[0] && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                        op->src[0]->ne[2] == op->src[1]->ne[1] && op->src[1]->ne[2] == op->src[2]->ne[1] &&
                        op->src[1]->ne[3] == op->src[2]->ne[2] && op->src[4]->ne[2] == op->src[1]->ne[2] &&
                        op->src[4]->ne[3] == op->src[1]->ne[3] && ggml_are_same_shape(op->src[4], op->src[5]) &&
                        op->src[6]->ne[0] == op->src[1]->ne[3] && op->src[3]->ne[1] == op->src[1]->ne[1] &&
                        (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[0]->ne[0]) &&
                        (op->src[1]->ne[1] % op->src[4]->ne[1] == 0);
            বিরতি;
        ক্ষেত্র GGML_OP_FLASH_ATTN_EXT:
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && op->src[2] &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == nullptr &&
                ggml_is_contiguous_rows(op) && ggml_is_contiguous_rows(op->src[0])) {
                দশমিক max_bias      = 0.0f;
                দশমিক logit_softcap = 0.0f;
                memcpy(&max_bias, (ধ্রুবক দশমিক *) op->op_params + 1, আকার(max_bias));
                memcpy(&logit_softcap, (ধ্রুবক দশমিক *) op->op_params + 2, আকার(logit_softcap));

                ধ্রুবক ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                যৌক্তিক mask_ok = (op->src[3] == nullptr) || (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                ধ্রুবক int64_t nhq = op->src[0]->ne[2];
                ধ্রুবক int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                ধ্রুবক size_t k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                ধ্রুবক size_t v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                // Only support matrix engine path (F16 K/V, dk%32==0);
                // mask scalar F32 fallback to get baseline perf readings
                ধ্রুবক যৌক্তিক me_eligible = op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16 &&
                                         (op->src[0]->ne[0] % 32) == 0;

                supported = me_eligible && mask_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                            max_bias == 0.0f && logit_softcap == 0.0f && op->src[0]->nb[0] == আকার(দশমিক) &&
                            op->src[1]->nb[0] == k_elem && op->src[2]->nb[0] == v_elem && op->nb[0] == আকার(দশমিক) &&
                            op->src[0]->ne[0] == op->src[1]->ne[0] &&  // dk matches
                            op->src[2]->ne[0] == op->ne[0] &&          // dv matches
                            op->src[2]->ne[0] <= 512 &&                // dv limit
                            op->src[0]->ne[0] <= 512 &&                // dk limit
                            nhq % nhk == 0 &&                          // GQA ratio is integer
                            op->src[0]->ne[1] == op->ne[2] && op->src[0]->ne[2] == op->ne[1] &&
                            op->src[0]->ne[3] == op->ne[3] && op->src[1]->ne[1] == op->src[2]->ne[1] &&
                            op->src[1]->ne[2] == op->src[2]->ne[2] && op->src[1]->ne[3] == op->src[2]->ne[3] &&
                            op->src[0]->ne[3] == op->src[1]->ne[3];
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_GET_ROWS:
            // Support F32/F16/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            যদি (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            যদি ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0] &&
                op->src[0]->type == op->type && ggml_is_contiguous(op)) {
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                যদি (op->তথ্য && op->src[0]->তথ্য && op->তথ্য == op->src[0]->তথ্য) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = মিথ্যা;
                } অন্যথা {
                    supported = সত্য;
                }
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            যদি (op->src[0]) {
                ধ্রুবক int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                যদি (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = সত্য;
                } অন্যথা যদি ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    যদি (op->তথ্য && op->src[0]->তথ্য && op->তথ্য == op->src[0]->তথ্য) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = মিথ্যা;
                    } অন্যথা {
                        supported = সত্য;
                    }
                } অন্যথা যদি (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = সত্য;
                } অন্যথা {
                    supported = মিথ্যা;
                }
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_CONCAT:
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                ধ্রুবক int32_t dim = ((ধ্রুবক int32_t *) op->op_params)[0];
                যদি (dim == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Fast dim==0 path: both source row segments are cacheline-aligned
                    // and contiguous, so the kernel can use vector row copies.
                    supported = সত্য;
                } অন্যথা যদি (dim == 0 && ((op->src[0]->nb[0] % আকার(দশমিক) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % আকার(দশমিক) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = সত্য;
                } অন্যথা যদি (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = সত্য;
                }
            }
            বিরতি;
        ক্ষেত্র GGML_OP_SSM_CONV:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == আকার(দশমিক) &&
                        op->src[1]->nb[0] == আকার(দশমিক) && op->src[0]->nb[1] == op->src[0]->ne[0] * আকার(দশমিক) &&
                        op->src[1]->nb[1] == op->src[1]->ne[0] * আকার(দশমিক) && ggml_is_contiguous(op) &&
                        op->src[1]->ne[1] == op->src[0]->ne[1] && op->ne[0] == op->src[0]->ne[1] &&
                        op->ne[1] == op->src[0]->ne[0] - op->src[1]->ne[0] + 1 && op->ne[2] == op->src[0]->ne[2];
            বিরতি;
        ক্ষেত্র GGML_OP_PAD:
            // F32 zero-pad only, no dim0 padding, dst contiguous
            // ne[0] must be CL-aligned (% 16 == 0) or evenly divide a CL (16 % ne[0] == 0)
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && (op->ne[0] % 16 == 0 || 16 % op->ne[0] == 0) &&
                op->src[0]->nb[0] == আকার(দশমিক)) {
                ধ্রুবক int32_t lp0      = ((ধ্রুবক int32_t *) op->op_params)[0];
                ধ্রুবক int32_t rp0      = ((ধ্রুবক int32_t *) op->op_params)[1];
                ধ্রুবক যৌক্তিক    circular = (যৌক্তিক) ((ধ্রুবক int32_t *) op->op_params)[8];
                যদি (lp0 == 0 && rp0 == 0 && !circular) {
                    supported = সত্য;
                } অন্যথা {
                    supported = মিথ্যা;
                }
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_REPEAT:
            // Two acceptable shapes:
            //   1. No-op REPEAT (src and dst have identical shape): dispatched
            //      to cont_f32, which handles arbitrary contiguous sizes.
            //   2. Real REPEAT via repeat_f32 kernel: dst ne[0] cacheline-aligned,
            //      src0 ne[0] cacheline-aligned or 1, dst.ne[i] % src0.ne[i] == 0.
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op)) {
                supported = সত্য;
            } অন্যথা যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = সত্য;
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_FILL:
            // F32 contiguous, ne[0] cacheline-aligned for SIMD fill
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] % 16 == 0;
            বিরতি;
        ক্ষেত্র GGML_OP_DIAG:
            // F32 contiguous dst, src0 is 1D vector [N,1,...], dst is [N,N,...]
            // ne[0] must be cacheline-aligned for SIMD zeroing
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && op->ne[0] == op->ne[1] && op->src[0]->ne[0] == op->ne[0] &&
                        op->src[0]->ne[1] == 1 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_TRI:
            // F32 contiguous, same shape in/out
            // Kernel handles arbitrary ne[0] with aligned fast path + scalar fallback
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            বিরতি;
        ক্ষেত্র GGML_OP_SOLVE_TRI:
            // F32 contiguous, A square, shapes compatible
            // Only lower-triangular left-side non-unit variant
            // Require k % 16 == 0 for cache-line-safe column parallelism
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == op->src[0]->ne[1] &&
                        op->src[0]->ne[1] == op->src[1]->ne[1] && op->src[1]->ne[0] % 16 == 0 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            বিরতি;
        ক্ষেত্র GGML_OP_SET:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                ধ্রুবক যৌক্তিক   inplace = (যৌক্তিক) ((ধ্রুবক int32_t *) op->op_params)[4];
                ধ্রুবক size_t nb1     = ((ধ্রুবক int32_t *) op->op_params)[0];
                ধ্রুবক size_t nb2     = ((ধ্রুবক int32_t *) op->op_params)[1];
                ধ্রুবক size_t nb3     = ((ধ্রুবক int32_t *) op->op_params)[2];
                ধ্রুবক size_t offset  = ((ধ্রুবক int32_t *) op->op_params)[3];
                ধ্রুবক size_t nb0     = ggml_element_আকার(op);
                ধ্রুবক size_t im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                ধ্রুবক size_t im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                ধ্রুবক size_t im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                ধ্রুবক size_t im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                ধ্রুবক যৌক্তিক view_bounds_ok = offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                ধ্রুবক যৌক্তিক cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (offset % 64 == 0);

                supported = inplace && view_bounds_ok && cacheline_aligned;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = সত্য;
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = সত্য;
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_GATED_DELTA_NET:
            // F32, S_v must be multiple of 8 for vectorization
            // q, k, v may be row-contiguous with strided higher dimensions.
            // g, beta, state stay contiguous.
            যদি (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&  // q
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
                supported = সত্য;
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_VIEW:
        ক্ষেত্র GGML_OP_PERMUTE:
        ক্ষেত্র GGML_OP_TRANSPOSE:
        ক্ষেত্র GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = সত্য;
            বিরতি;
        ক্ষেত্র GGML_OP_SET_ROWS:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            যদি (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
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
            } অন্যথা {
                supported = মিথ্যা;
            }
            বিরতি;
        ক্ষেত্র GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = সত্য;
            বিরতি;
        পূর্বনির্ধারিত:
            supported = মিথ্যা;
            বিরতি;
    }
    // if(!supported) {
    //     ggml_et_dump_operator_metadata(op);
    // }
    ফেরত supported;
}

স্থির যৌক্তিক ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_বাফার_type_t buft) {
    GGML_UNUSED(dev);
    ফেরত buft->iface.get_name == ggml_backend_et_বাফার_type_get_name;
}

স্থির যৌক্তিক ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, ধ্রুবক ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    যদি (op->op == GGML_OP_GET_ROWS) {
        ফেরত মিথ্যা;
    }
    ফেরত সত্য;

    GGML_UNUSED(dev);
}

স্থির ধ্রুবক কাঠামো ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ ফাঁকা,
    /* .get_tensor_2d_async     = */ ফাঁকা,
    /* .cpy_tensor_async        = */ ফাঁকা,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ ফাঁকা,
    /* .graph_plan_free         = */ ফাঁকা,
    /* .graph_plan_update       = */ ফাঁকা,
    /* .graph_plan_compute      = */ ফাঁকা,
    /* .graph_compute           = */ ggml_backend_et_graph_compute,
    /* .event_record            = */ ফাঁকা,
    /* .event_wait              = */ ফাঁকা,
    /* .graph_optimize          = */ ফাঁকা,
};

স্থির ধ্রুবক অক্ষর * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ফেরত dev_ctx->name.c_str();
}

স্থির ধ্রুবক অক্ষর * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ফেরত dev_ctx->desc.c_str();
}

স্থির শূন্য ggml_backend_et_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * যোগ) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *মুক্ত                                    = dev_ctx->যোগ_mem;
    *যোগ                                   = dev_ctx->যোগ_mem;
}

স্থির গণনা ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    ফেরত GGML_BACKEND_DEVICE_TYPE_GPU;
}

স্থির শূন্য ggml_backend_et_device_get_props(ggml_backend_dev_t dev, কাঠামো ggml_backend_dev_props * props) {
    GGML_UNUSED(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_memory(dev, &props->memory_free, &props->memory_যোগ);
    props->device_id = ফাঁকা;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ সত্য,
        /* .host_buffer           = */ মিথ্যা,
        /* .buffer_from_host_ptr  = */ মিথ্যা,
        /* .events                = */ মিথ্যা,
        /* .mmap_support          = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, ধ্রুবক অক্ষর * params) {
    GGML_UNUSED(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ফেরত ggml_backend_et_init(dev_ctx->devidx);
}

স্থির ggml_backend_বাফার_type_t ggml_backend_et_device_get_বাফার_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ফেরত dev_ctx->buftype;
}

স্থির ggml_backend_বাফার_type_t ggml_backend_et_device_get_host_বাফার_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    ফেরত ggml_backend_cpu_বাফার_type();
}

স্থির ধ্রুবক কাঠামো ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_memory,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_বাফার_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_বাফার_type,
    /* .buffer_from_host_ptr = */ ফাঁকা,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ ফাঁকা,
    /* .event_free        = */ ফাঁকা,
    /* .event_synchronize = */ ফাঁকা,
};

/*
  Backend Registry.
*/

স্থির ধ্রুবক অক্ষর * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    ফেরত GGML_ET_NAME;
}

স্থির size_t ggml_backend_et_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    ফেরত ctx->devices.size();
}

স্থির ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    যদি (devidx >= ctx->devices.size()) {
        ফেরত nullptr;
    }
    ফেরত ctx->devices[devidx];
}

স্থির শূন্য * ggml_backend_et_get_proc_address(ggml_backend_reg_t reg, ধ্রুবক অক্ষর * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    ফেরত nullptr;
}

স্থির ধ্রুবক কাঠামো ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_গুনতি,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_address,
};

ggml_backend_reg_t ggml_backend_et_reg(শূন্য) {
    স্থির ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = নতুন ggml_backend_et_reg_ctx;

        যদি (!ggml_et_driver_init()) {
            ফেরত nullptr;
        }

        ggml_backend_reg_t r = নতুন ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ nullptr,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        জন্য (পূর্ণসংখ্যা i = 0; i < ggml_et_devicelayer()->get_Devices_গুনতি(); i++) {
            ggml_backend_dev_t dev = নতুন ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ nullptr  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = নতুন ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->যোগ_mem                       = static_cast<size_t>(prop.memory_আকার);
            {
                ধ্রুবক অক্ষর * env            = getenv("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && strcmp(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_বাফার_type_context * bufty_ctx = নতুন ggml_backend_et_বাফার_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = নতুন ggml_backend_বাফার_type{ /* .iface   = */ ggml_backend_et_বাফার_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_stream = ggml_et_runtime()->createStream(rtid);

            dev_ctx->trace_বাফার = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_বাফার_আকার);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            জন্য (auto & slot : dev_ctx->uberkernel.slots) {
                slot.insts.reserve(256);
                slot.params_blob.reserve(1 << 20);
                slot.device_insts_capacity  = 256 * আকার(ggml_et_uberkernel_inst);
                slot.device_params_capacity = 1 << 20;
                slot.device_insts           = ggml_et_runtime()->mallocDevice(rtid, slot.device_insts_capacity);
                slot.device_params          = ggml_et_runtime()->mallocDevice(rtid, slot.device_params_capacity);
                যদি (slot.device_insts == nullptr) {
                    slot.device_insts_capacity = 0;
                }
                যদি (slot.device_params == nullptr) {
                    slot.device_params_capacity = 0;
                }
            }

            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
        }

        r->context = ctx;
        ফেরত r;
    }();

    ফেরত _reg;
}

ggml_guid_t ggml_backend_et_guid(শূন্য) {
    স্থির ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    ফেরত &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    যদি (!ggml_et_driver_init()) {
        ফেরত nullptr;
    }

    যদি (devidx >= (size_t) ggml_backend_et_get_device_গুনতি()) {
        ফেরত nullptr;
    }

    ggml_backend_et_context * ctx = নতুন ggml_backend_et_context;
    ctx->devidx                   = (পূর্ণসংখ্যা) devidx;

    ggml_backend_t backend = নতুন ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    ফেরত backend;
}

যৌক্তিক ggml_backend_is_et(ggml_backend_t backend) {
    ফেরত backend != ফাঁকা && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

পূর্ণসংখ্যা ggml_backend_et_get_device_গুনতি(শূন্য) {
    ফেরত ggml_backend_et_reg_get_device_গুনতি(ggml_backend_et_reg());
}

শূন্য ggml_backend_et_get_device_description(পূর্ণসংখ্যা devidx, অক্ষর * description, size_t description_আকার) {
    যদি (devidx < 0 || devidx >= ggml_backend_et_get_device_গুনতি()) {
        snprintf(description, description_আকার, "ET Device %d (invalid)", devidx);
        ফেরত;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_আকার, "%s", dev_ctx->desc.c_str());
}

শূন্য ggml_backend_et_get_device_memory(পূর্ণসংখ্যা devidx, size_t * মুক্ত, size_t * যোগ) {
    যদি (devidx < 0 || devidx >= ggml_backend_et_get_device_গুনতি()) {
        *মুক্ত  = 0;
        *যোগ = 0;
        ফেরত;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_memory(dev, মুক্ত, যোগ);
}

ggml_backend_বাফার_type_t ggml_backend_et_বাফার_type(size_t dev_অঙ্ক) {
    যদি (dev_অঙ্ক >= (size_t) ggml_backend_et_get_device_গুনতি()) {
        ফেরত nullptr;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_অঙ্ক);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ফেরত dev_ctx->buftype;
}

ggml_backend_বাফার_type_t ggml_backend_et_host_বাফার_type(শূন্য) {
    স্থির ggml_backend_বাফার_type host_বাফার_type = {
        /* .iface   = */ ggml_backend_et_বাফার_type_i,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    ফেরত &host_বাফার_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)

#ಅನ್ತರ್ಭಾವ "ggml-et-kernels.h"

#ಅನ್ತರ್ಭಾವ "ggml-et-kernels-embed.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-et-uberkernel-kernel-map.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <fstream>

#ವ್ಯಾಖ್ಯಾನಿಸು ET_TRACE_DECODER_IMPL
#ಅನ್ತರ್ಭಾವ <et-trace/decoder.h>
#ಅನ್ತರ್ಭಾವ <et-trace/layout.h>

ಸ್ಥಿರ constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

ಸ್ಥಿರ size_t ggml_et_align_ಮೇಲೆ(size_t value, size_t alignment) {
    ಹಿಂತಿರುಗಿ (value + alignment - 1) & ~(alignment - 1);
}

ಸ್ಥಿರ size_t ggml_et_ಮುಂದಿನ_ಸಾಮರ್ಥ್ಯ(size_t current_ಸಾಮರ್ಥ್ಯ, size_t required_ಸಾಮರ್ಥ್ಯ) {
    ಆದರೆ (current_ಸಾಮರ್ಥ್ಯ == 0) {
        ಹಿಂತಿರುಗಿ required_ಸಾಮರ್ಥ್ಯ;
    }

    size_t ಮುಂದಿನ_ಸಾಮರ್ಥ್ಯ = current_ಸಾಮರ್ಥ್ಯ;
    ಆಗ (ಮುಂದಿನ_ಸಾಮರ್ಥ್ಯ < required_ಸಾಮರ್ಥ್ಯ) {
        ಮುಂದಿನ_ಸಾಮರ್ಥ್ಯ *= 2;
    }

    ಹಿಂತಿರುಗಿ ಮುಂದಿನ_ಸಾಮರ್ಥ್ಯ;
}

ಸ್ಥಿರ ggml_backend_et_uberkernel_ಸ್ಥಾನ & ggml_et_uberkernel_current_ಸ್ಥಾನ(ggml_backend_et_uberkernel_context * uk_ctx) {
    ಹಿಂತಿರುಗಿ uk_ctx->slots[uk_ctx->current_ಸ್ಥಾನ];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_et_uberkernel_ಸ್ಥಾನ_wait(ggml_backend_et_uberkernel_ಸ್ಥಾನ &     ಸ್ಥಾನ,
                                         ಸ್ಥಿರಾಂಕ std::shared_ptr<rt::IRuntime> & runtime) {
    ಆದರೆ (!ಸ್ಥಾನ.has_pending || !runtime) {
        ಹಿಂತಿರುಗಿ;
    }
    runtime->waitForEvent(ಸ್ಥಾನ.pending_event);
    ಸ್ಥಾನ.has_pending = ಸುಳ್ಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    ಆದರೆ (!uk_ctx) {
        ಹಿಂತಿರುಗಿ;
    }

    uk_ctx->shire_ಮುಖವಾಡ = 0;
    ಸ್ವಯಂಚಾಲಿತ & ಸ್ಥಾನ        = ggml_et_uberkernel_current_ಸ್ಥಾನ(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_ಸ್ಥಾನ_wait(ಸ್ಥಾನ, ggml_et_runtime());
    ಸ್ಥಾನ.insts.clear();
    ಸ್ಥಾನ.params_blob.clear();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_uberkernel_ensure_ಸ್ಥಾನ_ಸಾಮರ್ಥ್ಯ(ggml_backend_et_uberkernel_ಸ್ಥಾನ & ಸ್ಥಾನ,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_ಗಾತ್ರ,
                                                    size_t                            params_ಗಾತ್ರ) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!dev_ctx || !runtime) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಪ್ರಯತ್ನಿಸು {
        ಆದರೆ (ಸ್ಥಾನ.device_insts == ಶೂನ್ಯಸೂಚಕ || insts_ಗಾತ್ರ > ಸ್ಥಾನ.device_insts_ಸಾಮರ್ಥ್ಯ) {
            ಸ್ಥಿರಾಂಕ size_t new_ಸಾಮರ್ಥ್ಯ = ggml_et_ಮುಂದಿನ_ಸಾಮರ್ಥ್ಯ(ಸ್ಥಾನ.device_insts_ಸಾಮರ್ಥ್ಯ, insts_ಗಾತ್ರ);
            ಆದರೆ (ಸ್ಥಾನ.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, ಸ್ಥಾನ.device_insts);
            }
            ಸ್ಥಾನ.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_ಸಾಮರ್ಥ್ಯ);
            ಸ್ಥಾನ.device_insts_ಸಾಮರ್ಥ್ಯ = ಸ್ಥಾನ.device_insts ? new_ಸಾಮರ್ಥ್ಯ : 0;
        }

        ಆದರೆ (ಸ್ಥಾನ.device_params == ಶೂನ್ಯಸೂಚಕ || params_ಗಾತ್ರ > ಸ್ಥಾನ.device_params_ಸಾಮರ್ಥ್ಯ) {
            ಸ್ಥಿರಾಂಕ size_t new_ಸಾಮರ್ಥ್ಯ = ggml_et_ಮುಂದಿನ_ಸಾಮರ್ಥ್ಯ(ಸ್ಥಾನ.device_params_ಸಾಮರ್ಥ್ಯ, params_ಗಾತ್ರ);
            ಆದರೆ (ಸ್ಥಾನ.device_params) {
                runtime->freeDevice(dev_ctx->rtid, ಸ್ಥಾನ.device_params);
            }
            ಸ್ಥಾನ.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_ಸಾಮರ್ಥ್ಯ);
            ಸ್ಥಾನ.device_params_ಸಾಮರ್ಥ್ಯ = ಸ್ಥಾನ.device_params ? new_ಸಾಮರ್ಥ್ಯ : 0;
        }
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ಸ್ಥಾನ.device_insts != ಶೂನ್ಯಸೂಚಕ && ಸ್ಥಾನ.device_params != ಶೂನ್ಯಸೂಚಕ;
}

// Get embedded kernel data by name
ಸ್ಥಿರ std::vector<std::byte> ggml_et_get_embedded_kernel(ಸ್ಥಿರಾಂಕ std::string & kernel_name) {
    ಸ್ವಯಂಚಾಲಿತ it = ggml_et_embedded_kernels.find(kernel_name);
    ಆದರೆ (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        ಹಿಂತಿರುಗಿ {};
    }

    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * ದತ್ತಾಂಶ = it->second.ಮೊದಲ;
    uint64_t              size = it->second.second;

    std::vector<std::byte> ಬಫರ್(size);
    std::ಸ್ಮರಣೆ_ನಕಲು(ಬಫರ್.ದತ್ತಾಂಶ(), ದತ್ತಾಂಶ, size);

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

// Read kernel from file (for development/override)
ಸ್ಥಿರ std::vector<std::byte> ggml_et_read_kernel_file(ಸ್ಥಿರಾಂಕ std::string & kernel_ಮಾರ್ಗ) {
    std::ifstream file(kernel_ಮಾರ್ಗ, std::ios::binary | std::ios::ate);
    ಆದರೆ (!file) {
        ಹಿಂತಿರುಗಿ {};
    }

    ಸ್ವಯಂಚಾಲಿತ size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> ಬಫರ್(size);
    file.ಪಠ(reinterpret_cast<ಅಕ್ಷರ *>(ಬಫರ್.ದತ್ತಾಂಶ()), size);

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

// Load kernel from file or embedded data
ತಾರ್ಕಿಕ ggml_et_ತುಂಬು_kernel(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Check if kernel already loaded
    ಆದರೆ (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    std::vector<std::byte> kernel_ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *           kernels_ಮಾರ್ಗ = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    ಆದರೆ (kernels_ಮಾರ್ಗ) {
        std::string kernel_file = std::string(kernels_ಮಾರ್ಗ) + "/" + kernel_name + ".elf";
        kernel_ದತ್ತಾಂಶ             = ggml_et_read_kernel_file(kernel_file);

        ಆದರೆ (!kernel_ದತ್ತಾಂಶ.ಖಾಲಿಗಳು()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    ಆದರೆ (kernel_ದತ್ತಾಂಶ.ಖಾಲಿಗಳು()) {
        kernel_ದತ್ತಾಂಶ = ggml_et_get_embedded_kernel(kernel_name);
        ಆದರೆ (kernel_ದತ್ತಾಂಶ.ಖಾಲಿಗಳು()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಪ್ರಯತ್ನಿಸು {
        // Load kernel code using device's default stream
        ಸ್ವಯಂಚಾಲಿತ ತುಂಬು_ಫಲಿತಾಂಶ = runtime->ತುಂಬು_ಸಂಕೇತ(dev_ctx->default_ಹರಿವು, kernel_ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), kernel_ದತ್ತಾಂಶ.size());
        runtime->waitForEvent(ತುಂಬು_ಫಲಿತಾಂಶ.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = ತುಂಬು_ಫಲಿತಾಂಶ.kernel_;
        ಹಿಂತಿರುಗಿ ನಿಜ;

    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_launch_kernel_ಆಂತರಿಕ(ggml_backend_et_device_context * dev_ctx,
                                           ಸ್ಥಿರಾಂಕ std::string &              kernel_name,
                                           ಶೂನ್ಯ *                           params,
                                           size_t                           params_ಗಾತ್ರ,
                                           uint64_t                         shire_ಮುಖವಾಡ,
                                           ತಾರ್ಕಿಕ                             enable_print,
                                           ತಾರ್ಕಿಕ                             sync_error_check,
                                           rt::EventId *                    ಹೊರಹೋಗು_event = ಶೂನ್ಯಸೂಚಕ) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    ಸ್ವಯಂಚಾಲಿತ kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ಆದರೆ (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        ಆದರೆ (!ggml_et_ತುಂಬು_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        ಆದರೆ (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    ಪ್ರಯತ್ನಿಸು {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.ಗಣ_Shire_ಮುಖವಾಡ(shire_ಮುಖವಾಡ);  // Default: all shires (0xFFFFFFFF)
        k_opts.ಗಣ_Barrier(ನಿಜ);          // Wait for completion
        k_opts.ಗಣ_Flush_L3(ಸುಳ್ಳು);         // No L3 flush needed
        ಆದರೆ (enable_print) {
            k_opts.ಗಣ_User_Tracing(reinterpret_cast<uint64_t>(dev_ctx->trace_ಬಫರ್),
                                  static_cast<uint32_t>(ET_TRACE_ಬಫರ್_ಗಾತ್ರ),
                                  0,                      //ಮಿತಿ
                                  shire_ಮುಖವಾಡ,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        ಆದರೆ (sync_error_check) {
            runtime->wait_For_ಹರಿವು(dev_ctx->default_ಹರಿವು);
            ಸ್ವಯಂಚಾಲಿತ errors = runtime->retrieve_ಹರಿವು_Errors(dev_ctx->default_ಹರಿವು);
            ಆದರೆ (!errors.ಖಾಲಿಗಳು()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (ಪೂರ್ಣಾಂಕ) error.error_ಸಂಕೇತ);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_ಹರಿವು, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_ಗಾತ್ರ, k_opts);
        ಆದರೆ (ಹೊರಹೋಗು_event) {
            *ಹೊರಹೋಗು_event = launch_event;
        }

        ಆದರೆ (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_ಬಫರ್_ಗಾತ್ರ);
            runtime->memcpy_Device_ಗಮ್ಯ_Host(dev_ctx->default_ಹರಿವು, dev_ctx->trace_ಬಫರ್, host_trace_buf.ದತ್ತಾಂಶ(),
                                        ET_TRACE_ಬಫರ್_ಗಾತ್ರ);
            runtime->wait_For_ಹರಿವು(dev_ctx->default_ಹರಿವು);
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * trace_ಶೀರ್ಷಿಕೆ = reinterpret_cast<ಸ್ಥಿರಾಂಕ trace_ಬಫರ್_std_ಶೀರ್ಷಿಕೆ_t *>(host_trace_buf.ದತ್ತಾಂಶ());
            ಸ್ಥಿರಾಂಕ trace_ನಮೂದು_ಶೀರ್ಷಿಕೆ_t * ನಮೂದು = ಶೂನ್ಯಸೂಚಕ;
            ಆಗ ((ನಮೂದು = Trace_Decode(trace_ಶೀರ್ಷಿಕೆ, ನಮೂದು))) {
                ಆದರೆ (ನಮೂದು->type != TRACE_TYPE_STRING) {
                    ಮುಂದುವರಿ;
                }
                ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * str_ನಮೂದು = reinterpret_cast<ಸ್ಥಿರಾಂಕ trace_string_t *>(ನಮೂದು);
                ಬರೆ("[hart %d] %s", ನಮೂದು->hart_id, str_ನಮೂದು->string);
            }
        }

        ಆದರೆ (sync_error_check) {
            // Already triggered. No need to retrigger
            ಆದರೆ (!enable_print) {
                runtime->wait_For_ಹರಿವು(dev_ctx->default_ಹರಿವು);
            }
            ಸ್ವಯಂಚಾಲಿತ errors = runtime->retrieve_ಹರಿವು_Errors(dev_ctx->default_ಹರಿವು);
            ಆದರೆ (!errors.ಖಾಲಿಗಳು()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (ಪೂರ್ಣಾಂಕ) error.error_ಸಂಕೇತ);
                }
                abort();
            }
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಶೂನ್ಯ ggml_et_uberkernel_begin_ನಕ್ಷೆ(ggml_backend_et_uberkernel_context * uk_ctx) {
    ಆದರೆ (!uk_ctx) {
        ಹಿಂತಿರುಗಿ;
    }

    uk_ctx->failed = ಸುಳ್ಳು;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    ಆದರೆ (!uk_ctx || !dev_ctx) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ವಯಂಚಾಲಿತ & ಸ್ಥಾನ = ggml_et_uberkernel_current_ಸ್ಥಾನ(uk_ctx);
    ಆದರೆ (ಸ್ಥಾನ.insts.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = ನಿಜ;
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ size_t   insts_ಗಾತ್ರ  = ಸ್ಥಾನ.insts.size() * ಗಾತ್ರ(ggml_et_uberkernel_inst);
    ಸ್ಥಿರಾಂಕ size_t   params_ಗಾತ್ರ = ಸ್ಥಾನ.params_blob.size();
    ಸ್ಥಿರಾಂಕ uint64_t shire_ಮುಖವಾಡ  = uk_ctx->shire_ಮುಖವಾಡ;
    ತಾರ್ಕಿಕ           ok          = ಸುಳ್ಳು;

    ಪ್ರಯತ್ನಿಸು {
        ಆದರೆ (!ggml_et_uberkernel_ensure_ಸ್ಥಾನ_ಸಾಮರ್ಥ್ಯ(ಸ್ಥಾನ, dev_ctx, insts_ಗಾತ್ರ, params_ಗಾತ್ರ)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = ನಿಜ;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            ಸ್ಥಾನ.insts.clear();
            ಸ್ಥಾನ.params_blob.clear();
            uk_ctx->shire_ಮುಖವಾಡ = 0;
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpy_Host_ಗಮ್ಯ_Device(dev_ctx->default_ಹರಿವು, reinterpret_cast<ಸ್ಥಿರಾಂಕ std::byte *>(ಸ್ಥಾನ.insts.ದತ್ತಾಂಶ()),
                                    ಸ್ಥಾನ.device_insts, insts_ಗಾತ್ರ, ನಿಜ);
        runtime->memcpy_Host_ಗಮ್ಯ_Device(dev_ctx->default_ಹರಿವು, ಸ್ಥಾನ.params_blob.ದತ್ತಾಂಶ(), ಸ್ಥಾನ.device_params, params_ಗಾತ್ರ,
                                    ನಿಜ);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(ಸ್ಥಾನ.insts.size()),
            static_cast<uint32_t>(ಗಾತ್ರ(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(ಸ್ಥಾನ.device_insts),
            reinterpret_cast<uint64_t>(ಸ್ಥಾನ.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_ಆಂತರಿಕ(dev_ctx, "uberkernel", &params, ಗಾತ್ರ(params), shire_ಮುಖವಾಡ, ಸುಳ್ಳು, ಸುಳ್ಳು,
                                            &launch_event);
        ಆದರೆ (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            ಸ್ಥಾನ.pending_event = launch_event;
            ಸ್ಥಾನ.has_pending   = ನಿಜ;
        }
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    ಆದರೆ (ok) {
        uk_ctx->current_ಸ್ಥಾನ = (uk_ctx->current_ಸ್ಥಾನ + 1) % ggml_backend_et_uberkernel_context::ಸ್ಥಾನ_ಎಣಿಕೆ_2;
        ಸ್ವಯಂಚಾಲಿತ & ಮುಂದಿನ          = ggml_et_uberkernel_current_ಸ್ಥಾನ(uk_ctx);
        ggml_et_uberkernel_ಸ್ಥಾನ_wait(ಮುಂದಿನ, runtime);
        ಮುಂದಿನ.insts.clear();
        ಮುಂದಿನ.params_blob.clear();
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಾನ.insts.clear();
        ಸ್ಥಾನ.params_blob.clear();
    }
    uk_ctx->shire_ಮುಖವಾಡ = 0;
    ಹಿಂತಿರುಗಿ ok;
}

ಶೂನ್ಯ ggml_et_uberkernel_abort_ನಕ್ಷೆ(ggml_backend_et_uberkernel_context * uk_ctx) {
    ಆದರೆ (!uk_ctx) {
        ಹಿಂತಿರುಗಿ;
    }

    uk_ctx->failed = ಸುಳ್ಳು;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ತಾರ್ಕಿಕ ggml_et_uberkernel_failed(ಸ್ಥಿರಾಂಕ ggml_backend_et_uberkernel_context * uk_ctx) {
    ಹಿಂತಿರುಗಿ uk_ctx && uk_ctx->failed;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ಸ್ಥಿರಾಂಕ std::string &              kernel_name,
                                      ಶೂನ್ಯ *                           params,
                                      size_t                           params_ಗಾತ್ರ,
                                      uint64_t                         shire_ಮುಖವಾಡ,
                                      ತಾರ್ಕಿಕ                             enable_print,
                                      ತಾರ್ಕಿಕ                             sync_error_check) {
    ಆದರೆ (!dev_ctx) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ಸ್ಥಿರಾಂಕ uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    ಆದರೆ (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        ಆದರೆ (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ ggml_et_launch_kernel_ಆಂತರಿಕ(dev_ctx, kernel_name, params, params_ಗಾತ್ರ, shire_ಮುಖವಾಡ, enable_print,
                                              sync_error_check);
    }

    ಸ್ವಯಂಚಾಲಿತ &       ಸ್ಥಾನ          = ggml_et_uberkernel_current_ಸ್ಥಾನ(uk_ctx);
    ಸ್ಥಿರಾಂಕ size_t params_ಸ್ಥಾನಾಂತರ = ggml_et_align_ಮೇಲೆ(ಸ್ಥಾನ.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    ಆದರೆ (params_ಸ್ಥಾನಾಂತರ > ಸ್ಥಾನ.params_blob.size()) {
        ಸ್ಥಾನ.params_blob.resize(params_ಸ್ಥಾನಾಂತರ);
    }

    ಸ್ಥಿರಾಂಕ std::byte * params_ಬೈಟುಗಳು = reinterpret_cast<ಸ್ಥಿರಾಂಕ std::byte *>(params);
    ಸ್ಥಾನ.params_blob.insert(ಸ್ಥಾನ.params_blob.end(), params_ಬೈಟುಗಳು, params_ಬೈಟುಗಳು + params_ಗಾತ್ರ);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_ಸ್ಥಾನಾಂತರ),
        static_cast<uint32_t>(params_ಗಾತ್ರ),
    };
    ಸ್ಥಾನ.insts.push_back(inst);

    ಆದರೆ (ಸ್ಥಾನ.insts.size() == 1) {
        uk_ctx->shire_ಮುಖವಾಡ = shire_ಮುಖವಾಡ;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_et_uberkernel_ಕೊನೆ_ನಕ್ಷೆ(ggml_backend_et_device_context * dev_ctx) {
    ಆದರೆ (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

ತಾರ್ಕಿಕ ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ಸ್ಥಿರಾಂಕ std::string &              kernel_name,
                           ಶೂನ್ಯ *                           params,
                           size_t                           params_ಗಾತ್ರ,
                           uint64_t                         shire_ಮುಖವಾಡ,
                           ತಾರ್ಕಿಕ                             enable_print,
                           ತಾರ್ಕಿಕ                             sync_error_check) {
    ಆದರೆ (!dev_ctx) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!dev_ctx->uberkernel_enabled) {
        ಹಿಂತಿರುಗಿ ggml_et_launch_kernel_ಆಂತರಿಕ(dev_ctx, kernel_name, params, params_ಗಾತ್ರ, shire_ಮುಖವಾಡ, enable_print,
                                              sync_error_check);
    }

    ಹಿಂತಿರುಗಿ ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_ಗಾತ್ರ, shire_ಮುಖವಾಡ, enable_print,
                                     sync_error_check);
}

ಶೂನ್ಯ ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ಸ್ಥಿರಾಂಕ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ವಯಂಚಾಲಿತ kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ಆದರೆ (kernel_it != dev_ctx->loaded_kernels.end()) {
        ಪ್ರಯತ್ನಿಸು {
            runtime->unload_ಸಂಕೇತ(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

ಶೂನ್ಯ ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    ಆದರೆ (!dev_ctx) {
        ಹಿಂತಿರುಗಿ;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_ಹೆಸರುಗಳು;
    kernel_ಹೆಸರುಗಳು.reserve(dev_ctx->loaded_kernels.size());
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_ಹೆಸರುಗಳು.push_back(kernel_pair.ಮೊದಲ);
    }

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kernel_name : kernel_ಹೆಸರುಗಳು) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    ಹಿಂತಿರುಗಿ loaded_kernels;
}

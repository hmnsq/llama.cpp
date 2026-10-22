#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-kernels.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-kernels-embed.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-uberkernel-kernel-map.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ET_TRACE_DECODER_IMPL
#ਅਨ੍ਤਰ੍ਭਾਵ <et-trace/decoder.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <et-trace/layout.h>

ਸਥਿਰ constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

ਸਥਿਰ size_t ggml_et_align_ਉੱਪਰ(size_t value, size_t alignment) {
    ਵਾਪਸ (value + alignment - 1) & ~(alignment - 1);
}

ਸਥਿਰ size_t ggml_et_ਅਗਲਾ_ਸਮਰੱਥਾ(size_t current_ਸਮਰੱਥਾ, size_t required_ਸਮਰੱਥਾ) {
    ਜੇ (current_ਸਮਰੱਥਾ == 0) {
        ਵਾਪਸ required_ਸਮਰੱਥਾ;
    }

    size_t ਅਗਲਾ_ਸਮਰੱਥਾ = current_ਸਮਰੱਥਾ;
    ਜਦੋਂ (ਅਗਲਾ_ਸਮਰੱਥਾ < required_ਸਮਰੱਥਾ) {
        ਅਗਲਾ_ਸਮਰੱਥਾ *= 2;
    }

    ਵਾਪਸ ਅਗਲਾ_ਸਮਰੱਥਾ;
}

ਸਥਿਰ ggml_backend_et_uberkernel_ਥਾਂ & ggml_et_uberkernel_current_ਥਾਂ(ggml_backend_et_uberkernel_context * uk_ctx) {
    ਵਾਪਸ uk_ctx->slots[uk_ctx->current_ਥਾਂ];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
ਸਥਿਰ ਖਾਲੀ ggml_et_uberkernel_ਥਾਂ_wait(ggml_backend_et_uberkernel_ਥਾਂ &     ਥਾਂ,
                                         ਅਚਲ std::shared_ptr<rt::IRuntime> & runtime) {
    ਜੇ (!ਥਾਂ.has_pending || !runtime) {
        ਵਾਪਸ;
    }
    runtime->waitForEvent(ਥਾਂ.pending_event);
    ਥਾਂ.has_pending = ਝੂਠ;
}

ਸਥਿਰ ਖਾਲੀ ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    ਜੇ (!uk_ctx) {
        ਵਾਪਸ;
    }

    uk_ctx->shire_ਨਕਾਬ = 0;
    ਸਵੈਚਾਲਿਤ & ਥਾਂ        = ggml_et_uberkernel_current_ਥਾਂ(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_ਥਾਂ_wait(ਥਾਂ, ggml_et_runtime());
    ਥਾਂ.insts.clear();
    ਥਾਂ.params_blob.clear();
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_et_uberkernel_ensure_ਥਾਂ_ਸਮਰੱਥਾ(ggml_backend_et_uberkernel_ਥਾਂ & ਥਾਂ,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_ਮਾਪ,
                                                    size_t                            params_ਮਾਪ) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!dev_ctx || !runtime) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਕੋਸ਼ਿਸ਼ {
        ਜੇ (ਥਾਂ.device_insts == ਖਾਲੀ_ਸੂਚਕ || insts_ਮਾਪ > ਥਾਂ.device_insts_ਸਮਰੱਥਾ) {
            ਅਚਲ size_t new_ਸਮਰੱਥਾ = ggml_et_ਅਗਲਾ_ਸਮਰੱਥਾ(ਥਾਂ.device_insts_ਸਮਰੱਥਾ, insts_ਮਾਪ);
            ਜੇ (ਥਾਂ.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, ਥਾਂ.device_insts);
            }
            ਥਾਂ.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_ਸਮਰੱਥਾ);
            ਥਾਂ.device_insts_ਸਮਰੱਥਾ = ਥਾਂ.device_insts ? new_ਸਮਰੱਥਾ : 0;
        }

        ਜੇ (ਥਾਂ.device_params == ਖਾਲੀ_ਸੂਚਕ || params_ਮਾਪ > ਥਾਂ.device_params_ਸਮਰੱਥਾ) {
            ਅਚਲ size_t new_ਸਮਰੱਥਾ = ggml_et_ਅਗਲਾ_ਸਮਰੱਥਾ(ਥਾਂ.device_params_ਸਮਰੱਥਾ, params_ਮਾਪ);
            ਜੇ (ਥਾਂ.device_params) {
                runtime->freeDevice(dev_ctx->rtid, ਥਾਂ.device_params);
            }
            ਥਾਂ.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_ਸਮਰੱਥਾ);
            ਥਾਂ.device_params_ਸਮਰੱਥਾ = ਥਾਂ.device_params ? new_ਸਮਰੱਥਾ : 0;
        }
    } ਫੜੋ (ਅਚਲ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਥਾਂ.device_insts != ਖਾਲੀ_ਸੂਚਕ && ਥਾਂ.device_params != ਖਾਲੀ_ਸੂਚਕ;
}

// Get embedded kernel data by name
ਸਥਿਰ std::vector<std::byte> ggml_et_get_embedded_kernel(ਅਚਲ std::string & kernel_name) {
    ਸਵੈਚਾਲਿਤ it = ggml_et_embedded_kernels.find(kernel_name);
    ਜੇ (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        ਵਾਪਸ {};
    }

    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ * ਅੰਕੜੇ = it->second.ਪਹਿਲਾ;
    uint64_t              size = it->second.second;

    std::vector<std::byte> ਬਫਰ(size);
    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਬਫਰ.ਅੰਕੜੇ(), ਅੰਕੜੇ, size);

    ਵਾਪਸ ਬਫਰ;
}

// Read kernel from file (for development/override)
ਸਥਿਰ std::vector<std::byte> ggml_et_read_kernel_file(ਅਚਲ std::string & kernel_ਰਸਤਾ) {
    std::ifstream file(kernel_ਰਸਤਾ, std::ios::binary | std::ios::ate);
    ਜੇ (!file) {
        ਵਾਪਸ {};
    }

    ਸਵੈਚਾਲਿਤ size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> ਬਫਰ(size);
    file.ਪਠ(reinterpret_cast<ਅੱਖਰ *>(ਬਫਰ.ਅੰਕੜੇ()), size);

    ਵਾਪਸ ਬਫਰ;
}

// Load kernel from file or embedded data
ਤਾਰਕਿਕ ggml_et_ਲੱਦੋ_kernel(ggml_backend_et_device_context * dev_ctx, ਅਚਲ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Check if kernel already loaded
    ਜੇ (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        ਵਾਪਸ ਸੱਚ;
    }

    std::vector<std::byte> kernel_ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ *           kernels_ਰਸਤਾ = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    ਜੇ (kernels_ਰਸਤਾ) {
        std::string kernel_file = std::string(kernels_ਰਸਤਾ) + "/" + kernel_name + ".elf";
        kernel_ਅੰਕੜੇ             = ggml_et_read_kernel_file(kernel_file);

        ਜੇ (!kernel_ਅੰਕੜੇ.ਖਾਲੀ_ਥਾਵਾਂ()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } ਨਹੀਂਤਾਂ {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    ਜੇ (kernel_ਅੰਕੜੇ.ਖਾਲੀ_ਥਾਵਾਂ()) {
        kernel_ਅੰਕੜੇ = ggml_et_get_embedded_kernel(kernel_name);
        ਜੇ (kernel_ਅੰਕੜੇ.ਖਾਲੀ_ਥਾਵਾਂ()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਕੋਸ਼ਿਸ਼ {
        // Load kernel code using device's default stream
        ਸਵੈਚਾਲਿਤ ਲੱਦੋ_ਨਤੀਜਾ = runtime->ਲੱਦੋ_ਕੋਡ(dev_ctx->default_ਧਾਰਾ, kernel_ਅੰਕੜੇ.ਅੰਕੜੇ(), kernel_ਅੰਕੜੇ.size());
        runtime->waitForEvent(ਲੱਦੋ_ਨਤੀਜਾ.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = ਲੱਦੋ_ਨਤੀਜਾ.kernel_;
        ਵਾਪਸ ਸੱਚ;

    } ਫੜੋ (ਅਚਲ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        ਵਾਪਸ ਝੂਠ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_et_launch_kernel_ਅੰਦਰੂਨੀ(ggml_backend_et_device_context * dev_ctx,
                                           ਅਚਲ std::string &              kernel_name,
                                           ਖਾਲੀ *                           params,
                                           size_t                           params_ਮਾਪ,
                                           uint64_t                         shire_ਨਕਾਬ,
                                           ਤਾਰਕਿਕ                             enable_print,
                                           ਤਾਰਕਿਕ                             sync_error_check,
                                           rt::EventId *                    ਬਾਹਰਲਾ_event = ਖਾਲੀ_ਸੂਚਕ) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        ਵਾਪਸ ਝੂਠ;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    ਸਵੈਚਾਲਿਤ kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ਜੇ (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        ਜੇ (!ggml_et_ਲੱਦੋ_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            ਵਾਪਸ ਝੂਠ;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        ਜੇ (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            ਵਾਪਸ ਝੂਠ;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    ਕੋਸ਼ਿਸ਼ {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.ਸੈੱਟ_Shire_ਨਕਾਬ(shire_ਨਕਾਬ);  // Default: all shires (0xFFFFFFFF)
        k_opts.ਸੈੱਟ_Barrier(ਸੱਚ);          // Wait for completion
        k_opts.ਸੈੱਟ_Flush_L3(ਝੂਠ);         // No L3 flush needed
        ਜੇ (enable_print) {
            k_opts.ਸੈੱਟ_User_Tracing(reinterpret_cast<uint64_t>(dev_ctx->trace_ਬਫਰ),
                                  static_cast<uint32_t>(ET_TRACE_ਬਫਰ_ਮਾਪ),
                                  0,                      //ਹੱਦ
                                  shire_ਨਕਾਬ,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        ਜੇ (sync_error_check) {
            runtime->wait_For_ਧਾਰਾ(dev_ctx->default_ਧਾਰਾ);
            ਸਵੈਚਾਲਿਤ errors = runtime->retrieve_ਧਾਰਾ_Errors(dev_ctx->default_ਧਾਰਾ);
            ਜੇ (!errors.ਖਾਲੀ_ਥਾਵਾਂ()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (ਪੂਰਨ) error.error_ਕੋਡ);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_ਧਾਰਾ, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_ਮਾਪ, k_opts);
        ਜੇ (ਬਾਹਰਲਾ_event) {
            *ਬਾਹਰਲਾ_event = launch_event;
        }

        ਜੇ (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_ਬਫਰ_ਮਾਪ);
            runtime->memcpy_Device_ਮੰਜ਼ਿਲ_Host(dev_ctx->default_ਧਾਰਾ, dev_ctx->trace_ਬਫਰ, host_trace_buf.ਅੰਕੜੇ(),
                                        ET_TRACE_ਬਫਰ_ਮਾਪ);
            runtime->wait_For_ਧਾਰਾ(dev_ctx->default_ਧਾਰਾ);
            ਅਚਲ ਸਵੈਚਾਲਿਤ * trace_ਸਿਰਲੇਖ = reinterpret_cast<ਅਚਲ trace_ਬਫਰ_std_ਸਿਰਲੇਖ_t *>(host_trace_buf.ਅੰਕੜੇ());
            ਅਚਲ trace_ਇੰਦਰਾਜ_ਸਿਰਲੇਖ_t * ਇੰਦਰਾਜ = ਖਾਲੀ_ਸੂਚਕ;
            ਜਦੋਂ ((ਇੰਦਰਾਜ = Trace_Decode(trace_ਸਿਰਲੇਖ, ਇੰਦਰਾਜ))) {
                ਜੇ (ਇੰਦਰਾਜ->type != TRACE_TYPE_STRING) {
                    ਜਾਰੀ;
                }
                ਅਚਲ ਸਵੈਚਾਲਿਤ * str_ਇੰਦਰਾਜ = reinterpret_cast<ਅਚਲ trace_string_t *>(ਇੰਦਰਾਜ);
                ਲਿਖੋ("[hart %d] %s", ਇੰਦਰਾਜ->hart_id, str_ਇੰਦਰਾਜ->string);
            }
        }

        ਜੇ (sync_error_check) {
            // Already triggered. No need to retrigger
            ਜੇ (!enable_print) {
                runtime->wait_For_ਧਾਰਾ(dev_ctx->default_ਧਾਰਾ);
            }
            ਸਵੈਚਾਲਿਤ errors = runtime->retrieve_ਧਾਰਾ_Errors(dev_ctx->default_ਧਾਰਾ);
            ਜੇ (!errors.ਖਾਲੀ_ਥਾਵਾਂ()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (ਪੂਰਨ) error.error_ਕੋਡ);
                }
                abort();
            }
        }

        ਵਾਪਸ ਸੱਚ;
    } ਫੜੋ (ਅਚਲ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        ਵਾਪਸ ਝੂਠ;
    }
}

ਖਾਲੀ ggml_et_uberkernel_begin_ਗ੍ਰਾਫ(ggml_backend_et_uberkernel_context * uk_ctx) {
    ਜੇ (!uk_ctx) {
        ਵਾਪਸ;
    }

    uk_ctx->failed = ਝੂਠ;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    ਜੇ (!uk_ctx || !dev_ctx) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਸਵੈਚਾਲਿਤ & ਥਾਂ = ggml_et_uberkernel_current_ਥਾਂ(uk_ctx);
    ਜੇ (ਥਾਂ.insts.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ ਸੱਚ;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = ਸੱਚ;
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ size_t   insts_ਮਾਪ  = ਥਾਂ.insts.size() * ਆਕਾਰ(ggml_et_uberkernel_inst);
    ਅਚਲ size_t   params_ਮਾਪ = ਥਾਂ.params_blob.size();
    ਅਚਲ uint64_t shire_ਨਕਾਬ  = uk_ctx->shire_ਨਕਾਬ;
    ਤਾਰਕਿਕ           ok          = ਝੂਠ;

    ਕੋਸ਼ਿਸ਼ {
        ਜੇ (!ggml_et_uberkernel_ensure_ਥਾਂ_ਸਮਰੱਥਾ(ਥਾਂ, dev_ctx, insts_ਮਾਪ, params_ਮਾਪ)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = ਸੱਚ;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            ਥਾਂ.insts.clear();
            ਥਾਂ.params_blob.clear();
            uk_ctx->shire_ਨਕਾਬ = 0;
            ਵਾਪਸ ਝੂਠ;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpy_Host_ਮੰਜ਼ਿਲ_Device(dev_ctx->default_ਧਾਰਾ, reinterpret_cast<ਅਚਲ std::byte *>(ਥਾਂ.insts.ਅੰਕੜੇ()),
                                    ਥਾਂ.device_insts, insts_ਮਾਪ, ਸੱਚ);
        runtime->memcpy_Host_ਮੰਜ਼ਿਲ_Device(dev_ctx->default_ਧਾਰਾ, ਥਾਂ.params_blob.ਅੰਕੜੇ(), ਥਾਂ.device_params, params_ਮਾਪ,
                                    ਸੱਚ);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(ਥਾਂ.insts.size()),
            static_cast<uint32_t>(ਆਕਾਰ(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(ਥਾਂ.device_insts),
            reinterpret_cast<uint64_t>(ਥਾਂ.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_ਅੰਦਰੂਨੀ(dev_ctx, "uberkernel", &params, ਆਕਾਰ(params), shire_ਨਕਾਬ, ਝੂਠ, ਝੂਠ,
                                            &launch_event);
        ਜੇ (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            ਥਾਂ.pending_event = launch_event;
            ਥਾਂ.has_pending   = ਸੱਚ;
        }
    } ਫੜੋ (ਅਚਲ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    ਜੇ (ok) {
        uk_ctx->current_ਥਾਂ = (uk_ctx->current_ਥਾਂ + 1) % ggml_backend_et_uberkernel_context::ਥਾਂ_ਗਿਣਤੀ_2;
        ਸਵੈਚਾਲਿਤ & ਅਗਲਾ          = ggml_et_uberkernel_current_ਥਾਂ(uk_ctx);
        ggml_et_uberkernel_ਥਾਂ_wait(ਅਗਲਾ, runtime);
        ਅਗਲਾ.insts.clear();
        ਅਗਲਾ.params_blob.clear();
    } ਨਹੀਂਤਾਂ {
        ਥਾਂ.insts.clear();
        ਥਾਂ.params_blob.clear();
    }
    uk_ctx->shire_ਨਕਾਬ = 0;
    ਵਾਪਸ ok;
}

ਖਾਲੀ ggml_et_uberkernel_abort_ਗ੍ਰਾਫ(ggml_backend_et_uberkernel_context * uk_ctx) {
    ਜੇ (!uk_ctx) {
        ਵਾਪਸ;
    }

    uk_ctx->failed = ਝੂਠ;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ਤਾਰਕਿਕ ggml_et_uberkernel_failed(ਅਚਲ ggml_backend_et_uberkernel_context * uk_ctx) {
    ਵਾਪਸ uk_ctx && uk_ctx->failed;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ਅਚਲ std::string &              kernel_name,
                                      ਖਾਲੀ *                           params,
                                      size_t                           params_ਮਾਪ,
                                      uint64_t                         shire_ਨਕਾਬ,
                                      ਤਾਰਕਿਕ                             enable_print,
                                      ਤਾਰਕਿਕ                             sync_error_check) {
    ਜੇ (!dev_ctx) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ਅਚਲ uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    ਜੇ (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        ਜੇ (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ ggml_et_launch_kernel_ਅੰਦਰੂਨੀ(dev_ctx, kernel_name, params, params_ਮਾਪ, shire_ਨਕਾਬ, enable_print,
                                              sync_error_check);
    }

    ਸਵੈਚਾਲਿਤ &       ਥਾਂ          = ggml_et_uberkernel_current_ਥਾਂ(uk_ctx);
    ਅਚਲ size_t params_ਵਿੱਥ = ggml_et_align_ਉੱਪਰ(ਥਾਂ.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    ਜੇ (params_ਵਿੱਥ > ਥਾਂ.params_blob.size()) {
        ਥਾਂ.params_blob.resize(params_ਵਿੱਥ);
    }

    ਅਚਲ std::byte * params_ਬਾਈਟਾਂ = reinterpret_cast<ਅਚਲ std::byte *>(params);
    ਥਾਂ.params_blob.insert(ਥਾਂ.params_blob.end(), params_ਬਾਈਟਾਂ, params_ਬਾਈਟਾਂ + params_ਮਾਪ);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_ਵਿੱਥ),
        static_cast<uint32_t>(params_ਮਾਪ),
    };
    ਥਾਂ.insts.push_back(inst);

    ਜੇ (ਥਾਂ.insts.size() == 1) {
        uk_ctx->shire_ਨਕਾਬ = shire_ਨਕਾਬ;
    }

    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_et_uberkernel_ਸਿਰਾ_ਗ੍ਰਾਫ(ggml_backend_et_device_context * dev_ctx) {
    ਜੇ (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਵਾਪਸ ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

ਤਾਰਕਿਕ ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ਅਚਲ std::string &              kernel_name,
                           ਖਾਲੀ *                           params,
                           size_t                           params_ਮਾਪ,
                           uint64_t                         shire_ਨਕਾਬ,
                           ਤਾਰਕਿਕ                             enable_print,
                           ਤਾਰਕਿਕ                             sync_error_check) {
    ਜੇ (!dev_ctx) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!dev_ctx->uberkernel_enabled) {
        ਵਾਪਸ ggml_et_launch_kernel_ਅੰਦਰੂਨੀ(dev_ctx, kernel_name, params, params_ਮਾਪ, shire_ਨਕਾਬ, enable_print,
                                              sync_error_check);
    }

    ਵਾਪਸ ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_ਮਾਪ, shire_ਨਕਾਬ, enable_print,
                                     sync_error_check);
}

ਖਾਲੀ ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ਅਚਲ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        ਵਾਪਸ;
    }

    ਸਵੈਚਾਲਿਤ kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ਜੇ (kernel_it != dev_ctx->loaded_kernels.end()) {
        ਕੋਸ਼ਿਸ਼ {
            runtime->unload_ਕੋਡ(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } ਫੜੋ (ਅਚਲ std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

ਖਾਲੀ ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    ਜੇ (!dev_ctx) {
        ਵਾਪਸ;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_ਨਾਂ;
    kernel_ਨਾਂ.reserve(dev_ctx->loaded_kernels.size());
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_ਨਾਂ.push_back(kernel_pair.ਪਹਿਲਾ);
    }

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kernel_name : kernel_ਨਾਂ) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    ਵਾਪਸ loaded_kernels;
}

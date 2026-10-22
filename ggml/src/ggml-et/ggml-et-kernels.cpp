#ಅನ್ತರ್ಭಾವ "ggml-et-kernels.h"

#ಅನ್ತರ್ಭಾವ "ggml-et-kernels-embed.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-et-uberkernel-kernel-map.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <fstream>

#define ET_TRACE_DECODER_IMPL
#ಅನ್ತರ್ಭಾವ <et-trace/decoder.h>
#ಅನ್ತರ್ಭಾವ <et-trace/layout.h>

ಸ್ಥಿರ constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

ಸ್ಥಿರ size_t ggml_et_align_up(size_t value, size_t alignment) {
    ಪ್ರತ್ಯಾಗಮ (value + alignment - 1) & ~(alignment - 1);
}

ಸ್ಥಿರ size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    ಯದಿ (current_capacity == 0) {
        ಪ್ರತ್ಯಾಗಮ required_capacity;
    }

    size_t next_capacity = current_capacity;
    ಯಾವತ್ (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    ಪ್ರತ್ಯಾಗಮ next_capacity;
}

ಸ್ಥಿರ ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    ಪ್ರತ್ಯಾಗಮ uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
ಸ್ಥಿರ ಶೂನ್ಯ ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         ಧ್ರುವ std::shared_ptr<rt::IRuntime> & runtime) {
    ಯದಿ (!slot.has_pending || !runtime) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = ಅಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    ಯದಿ (!uk_ctx) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    uk_ctx->shire_mask = 0;
    auto & slot        = ggml_et_uberkernel_current_slot(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_slot_wait(slot, ggml_et_runtime());
    slot.insts.clear();
    slot.params_blob.clear();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_size,
                                                    size_t                            params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಯದಿ (!dev_ctx || !runtime) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯತ್ನ {
        ಯದಿ (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            ಧ್ರುವ size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            ಯದಿ (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        ಯದಿ (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            ಧ್ರುವ size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            ಯದಿ (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
ಸ್ಥಿರ std::vector<std::byte> ggml_et_get_embedded_kernel(ಧ್ರುವ std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    ಯದಿ (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        ಪ್ರತ್ಯಾಗಮ {};
    }

    ಧ್ರುವ unsigned ವರ್ಣ * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    ಪ್ರತ್ಯಾಗಮ buffer;
}

// Read kernel from file (for development/override)
ಸ್ಥಿರ std::vector<std::byte> ggml_et_read_kernel_file(ಧ್ರುವ std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    ಯದಿ (!file) {
        ಪ್ರತ್ಯಾಗಮ {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.ಪಠ(reinterpret_cast<ವರ್ಣ *>(buffer.data()), size);

    ಪ್ರತ್ಯಾಗಮ buffer;
}

// Load kernel from file or embedded data
ತಾರ್ಕಿಕ ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, ಧ್ರುವ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಯದಿ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Check if kernel already loaded
    ಯದಿ (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    std::vector<std::byte> kernel_data;
    ಧ್ರುವ ವರ್ಣ *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    ಯದಿ (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        ಯದಿ (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } ಅನ್ಯಥಾ {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    ಯದಿ (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        ಯದಿ (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಯತ್ನ {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

    } ಗೃಹಾಣ (ಧ್ರುವ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           ಧ್ರುವ std::string &              kernel_name,
                                           ಶೂನ್ಯ *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           ತಾರ್ಕಿಕ                             enable_print,
                                           ತಾರ್ಕಿಕ                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಯದಿ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ಯದಿ (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        ಯದಿ (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        ಯದಿ (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    ಯತ್ನ {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(ಸತ್ಯ);          // Wait for completion
        k_opts.setFlushL3(ಅಸತ್ಯ);         // No L3 flush needed
        ಯದಿ (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                                  static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        ಯದಿ (sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            ಯದಿ (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                ಪ್ರತಿ (ಧ್ರುವ auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (ಪೂರ್ಣಾಙ್ಕ) error.errorCode_);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_size, k_opts);
        ಯದಿ (out_event) {
            *out_event = launch_event;
        }

        ಯದಿ (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_buffer, host_trace_buf.data(),
                                        ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            ಧ್ರುವ auto * trace_header = reinterpret_cast<ಧ್ರುವ trace_buffer_std_header_t *>(host_trace_buf.data());
            ಧ್ರುವ trace_entry_header_t * entry = nullptr;
            ಯಾವತ್ ((entry = Trace_Decode(trace_header, entry))) {
                ಯದಿ (entry->type != TRACE_TYPE_STRING) {
                    ಅನುವರ್ತ;
                }
                ಧ್ರುವ auto * str_entry = reinterpret_cast<ಧ್ರುವ trace_string_t *>(entry);
                printf("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        ಯದಿ (sync_error_check) {
            // Already triggered. No need to retrigger
            ಯದಿ (!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            ಯದಿ (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                ಪ್ರತಿ (ಧ್ರುವ auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (ಪೂರ್ಣಾಙ್ಕ) error.errorCode_);
                }
                abort();
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

ಶೂನ್ಯ ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    ಯದಿ (!uk_ctx) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    uk_ctx->failed = ಅಸತ್ಯ;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    ಯದಿ (!uk_ctx || !dev_ctx) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    ಯದಿ (slot.insts.empty()) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಯದಿ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = ಸತ್ಯ;
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ size_t   insts_size  = slot.insts.size() * ಪರಿಮಾಣ(ggml_et_uberkernel_inst);
    ಧ್ರುವ size_t   params_size = slot.params_blob.size();
    ಧ್ರುವ uint64_t shire_mask  = uk_ctx->shire_mask;
    ತಾರ್ಕಿಕ           ok          = ಅಸತ್ಯ;

    ಯತ್ನ {
        ಯದಿ (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = ಸತ್ಯ;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<ಧ್ರುವ std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, ಸತ್ಯ);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    ಸತ್ಯ);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(ಪರಿಮಾಣ(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, ಪರಿಮಾಣ(params), shire_mask, ಅಸತ್ಯ, ಅಸತ್ಯ,
                                            &launch_event);
        ಯದಿ (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = ಸತ್ಯ;
        }
    } ಗೃಹಾಣ (ಧ್ರುವ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    ಯದಿ (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_ಎಣಿಕೆ;
        auto & next          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(next, runtime);
        next.insts.clear();
        next.params_blob.clear();
    } ಅನ್ಯಥಾ {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    ಪ್ರತ್ಯಾಗಮ ok;
}

ಶೂನ್ಯ ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    ಯದಿ (!uk_ctx) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    uk_ctx->failed = ಅಸತ್ಯ;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ತಾರ್ಕಿಕ ggml_et_uberkernel_failed(ಧ್ರುವ ggml_backend_et_uberkernel_context * uk_ctx) {
    ಪ್ರತ್ಯಾಗಮ uk_ctx && uk_ctx->failed;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ಧ್ರುವ std::string &              kernel_name,
                                      ಶೂನ್ಯ *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      ತಾರ್ಕಿಕ                             enable_print,
                                      ತಾರ್ಕಿಕ                             sync_error_check) {
    ಯದಿ (!dev_ctx) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ಧ್ರುವ uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    ಯದಿ (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        ಯದಿ (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    ಧ್ರುವ size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    ಯದಿ (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    ಧ್ರುವ std::byte * params_bytes = reinterpret_cast<ಧ್ರುವ std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    slot.insts.push_back(inst);

    ಯದಿ (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    ಯದಿ (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

ತಾರ್ಕಿಕ ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ಧ್ರುವ std::string &              kernel_name,
                           ಶೂನ್ಯ *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           ತಾರ್ಕಿಕ                             enable_print,
                           ತಾರ್ಕಿಕ                             sync_error_check) {
    ಯದಿ (!dev_ctx) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!dev_ctx->uberkernel_enabled) {
        ಪ್ರತ್ಯಾಗಮ ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    ಪ್ರತ್ಯಾಗಮ ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

ಶೂನ್ಯ ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ಧ್ರುವ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಯದಿ (!runtime) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ಯದಿ (kernel_it != dev_ctx->loaded_kernels.end()) {
        ಯತ್ನ {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } ಗೃಹಾಣ (ಧ್ರುವ std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

ಶೂನ್ಯ ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    ಯದಿ (!dev_ctx) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    ಪ್ರತಿ (ಧ್ರುವ auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    ಪ್ರತಿ (ಧ್ರುವ auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    ಪ್ರತಿ (ಧ್ರುವ auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    ಪ್ರತ್ಯಾಗಮ loaded_kernels;
}

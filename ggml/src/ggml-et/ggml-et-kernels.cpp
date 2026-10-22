#अन्तर्भाव "ggml-et-kernels.h"

#अन्तर्भाव "ggml-et-kernels-embed.hpp"
#अन्तर्भाव "ggml-et-uberkernel-kernel-map.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <fstream>

#define ET_TRACE_DECODER_IMPL
#अन्तर्भाव <et-trace/decoder.h>
#अन्तर्भाव <et-trace/layout.h>

स्थिर constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

स्थिर size_t ggml_et_align_up(size_t value, size_t alignment) {
    प्रत्यागम (value + alignment - 1) & ~(alignment - 1);
}

स्थिर size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    यदि (current_capacity == 0) {
        प्रत्यागम required_capacity;
    }

    size_t next_capacity = current_capacity;
    यावत् (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    प्रत्यागम next_capacity;
}

स्थिर ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    प्रत्यागम uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
स्थिर शून्य ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         ध्रुव std::shared_ptr<rt::IRuntime> & runtime) {
    यदि (!slot.has_pending || !runtime) {
        प्रत्यागम;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = असत्य;
}

स्थिर शून्य ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    यदि (!uk_ctx) {
        प्रत्यागम;
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

स्थिर तार्किक ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_size,
                                                    size_t                            params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!dev_ctx || !runtime) {
        प्रत्यागम असत्य;
    }

    यत्न {
        यदि (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            ध्रुव size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            यदि (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        यदि (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            ध्रुव size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            यदि (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } गृहाण (ध्रुव std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        प्रत्यागम असत्य;
    }

    प्रत्यागम slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
स्थिर std::vector<std::byte> ggml_et_get_embedded_kernel(ध्रुव std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    यदि (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        प्रत्यागम {};
    }

    ध्रुव unsigned वर्ण * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    प्रत्यागम buffer;
}

// Read kernel from file (for development/override)
स्थिर std::vector<std::byte> ggml_et_read_kernel_file(ध्रुव std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    यदि (!file) {
        प्रत्यागम {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.पठ(reinterpret_cast<वर्ण *>(buffer.data()), size);

    प्रत्यागम buffer;
}

// Load kernel from file or embedded data
तार्किक ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, ध्रुव std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        प्रत्यागम असत्य;
    }

    // Check if kernel already loaded
    यदि (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        प्रत्यागम सत्य;
    }

    std::vector<std::byte> kernel_data;
    ध्रुव वर्ण *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    यदि (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        यदि (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } अन्यथा {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    यदि (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        यदि (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            प्रत्यागम असत्य;
        }
    }

    यत्न {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        प्रत्यागम सत्य;

    } गृहाण (ध्रुव std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        प्रत्यागम असत्य;
    }
}

स्थिर तार्किक ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           ध्रुव std::string &              kernel_name,
                                           शून्य *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           तार्किक                             enable_print,
                                           तार्किक                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        प्रत्यागम असत्य;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    यदि (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        यदि (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            प्रत्यागम असत्य;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        यदि (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            प्रत्यागम असत्य;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    यत्न {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(सत्य);          // Wait for completion
        k_opts.setFlushL3(असत्य);         // No L3 flush needed
        यदि (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                                  static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        यदि (sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            यदि (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                प्रति (ध्रुव auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (पूर्णाङ्क) error.errorCode_);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_size, k_opts);
        यदि (out_event) {
            *out_event = launch_event;
        }

        यदि (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_buffer, host_trace_buf.data(),
                                        ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            ध्रुव auto * trace_header = reinterpret_cast<ध्रुव trace_buffer_std_header_t *>(host_trace_buf.data());
            ध्रुव trace_entry_header_t * entry = nullptr;
            यावत् ((entry = Trace_Decode(trace_header, entry))) {
                यदि (entry->type != TRACE_TYPE_STRING) {
                    अनुवर्त;
                }
                ध्रुव auto * str_entry = reinterpret_cast<ध्रुव trace_string_t *>(entry);
                printf("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        यदि (sync_error_check) {
            // Already triggered. No need to retrigger
            यदि (!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            यदि (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                प्रति (ध्रुव auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (पूर्णाङ्क) error.errorCode_);
                }
                abort();
            }
        }

        प्रत्यागम सत्य;
    } गृहाण (ध्रुव std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        प्रत्यागम असत्य;
    }
}

शून्य ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    यदि (!uk_ctx) {
        प्रत्यागम;
    }

    uk_ctx->failed = असत्य;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

स्थिर तार्किक ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    यदि (!uk_ctx || !dev_ctx) {
        प्रत्यागम असत्य;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    यदि (slot.insts.empty()) {
        प्रत्यागम सत्य;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = सत्य;
        प्रत्यागम असत्य;
    }

    ध्रुव size_t   insts_size  = slot.insts.size() * परिमाण(ggml_et_uberkernel_inst);
    ध्रुव size_t   params_size = slot.params_blob.size();
    ध्रुव uint64_t shire_mask  = uk_ctx->shire_mask;
    तार्किक           ok          = असत्य;

    यत्न {
        यदि (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = सत्य;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            प्रत्यागम असत्य;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<ध्रुव std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, सत्य);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    सत्य);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(परिमाण(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, परिमाण(params), shire_mask, असत्य, असत्य,
                                            &launch_event);
        यदि (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = सत्य;
        }
    } गृहाण (ध्रुव std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    यदि (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_गिनती;
        auto & next          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(next, runtime);
        next.insts.clear();
        next.params_blob.clear();
    } अन्यथा {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    प्रत्यागम ok;
}

शून्य ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    यदि (!uk_ctx) {
        प्रत्यागम;
    }

    uk_ctx->failed = असत्य;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

तार्किक ggml_et_uberkernel_failed(ध्रुव ggml_backend_et_uberkernel_context * uk_ctx) {
    प्रत्यागम uk_ctx && uk_ctx->failed;
}

स्थिर तार्किक ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ध्रुव std::string &              kernel_name,
                                      शून्य *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      तार्किक                             enable_print,
                                      तार्किक                             sync_error_check) {
    यदि (!dev_ctx) {
        प्रत्यागम असत्य;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ध्रुव uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    यदि (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        यदि (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            प्रत्यागम असत्य;
        }
        प्रत्यागम ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    ध्रुव size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    यदि (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    ध्रुव std::byte * params_bytes = reinterpret_cast<ध्रुव std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    slot.insts.push_back(inst);

    यदि (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    प्रत्यागम सत्य;
}

तार्किक ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    यदि (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        प्रत्यागम सत्य;
    }

    प्रत्यागम ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

तार्किक ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ध्रुव std::string &              kernel_name,
                           शून्य *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           तार्किक                             enable_print,
                           तार्किक                             sync_error_check) {
    यदि (!dev_ctx) {
        प्रत्यागम असत्य;
    }

    यदि (!dev_ctx->uberkernel_enabled) {
        प्रत्यागम ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    प्रत्यागम ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

शून्य ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ध्रुव std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        प्रत्यागम;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    यदि (kernel_it != dev_ctx->loaded_kernels.end()) {
        यत्न {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } गृहाण (ध्रुव std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

शून्य ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    यदि (!dev_ctx) {
        प्रत्यागम;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    प्रति (ध्रुव auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    प्रति (ध्रुव auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    प्रति (ध्रुव auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    प्रत्यागम loaded_kernels;
}

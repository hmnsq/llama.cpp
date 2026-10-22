#અન્તર્ભાવ "ggml-et-kernels.h"

#અન્તર્ભાવ "ggml-et-kernels-embed.hpp"
#અન્તર્ભાવ "ggml-et-uberkernel-kernel-map.h"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <fstream>

#define ET_TRACE_DECODER_IMPL
#અન્તર્ભાવ <et-trace/decoder.h>
#અન્તર્ભાવ <et-trace/layout.h>

સ્થિર constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

સ્થિર size_t ggml_et_align_up(size_t value, size_t alignment) {
    પાછા (value + alignment - 1) & ~(alignment - 1);
}

સ્થિર size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    જો (current_capacity == 0) {
        પાછા required_capacity;
    }

    size_t next_capacity = current_capacity;
    જ્યારે (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    પાછા next_capacity;
}

સ્થિર ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    પાછા uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
સ્થિર શૂન્ય ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         અચલ std::shared_ptr<rt::IRuntime> & runtime) {
    જો (!slot.has_pending || !runtime) {
        પાછા;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = ખોટું;
}

સ્થિર શૂન્ય ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    જો (!uk_ctx) {
        પાછા;
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

સ્થિર તાર્કિક ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_size,
                                                    size_t                            params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    જો (!dev_ctx || !runtime) {
        પાછા ખોટું;
    }

    પ્રયાસ {
        જો (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            અચલ size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            જો (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        જો (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            અચલ size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            જો (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } પકડો (અચલ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        પાછા ખોટું;
    }

    પાછા slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
સ્થિર std::vector<std::byte> ggml_et_get_embedded_kernel(અચલ std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    જો (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        પાછા {};
    }

    અચલ unsigned અક્ષર * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    પાછા buffer;
}

// Read kernel from file (for development/override)
સ્થિર std::vector<std::byte> ggml_et_read_kernel_file(અચલ std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    જો (!file) {
        પાછા {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.પઠ(reinterpret_cast<અક્ષર *>(buffer.data()), size);

    પાછા buffer;
}

// Load kernel from file or embedded data
તાર્કિક ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, અચલ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    જો (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        પાછા ખોટું;
    }

    // Check if kernel already loaded
    જો (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        પાછા સાચું;
    }

    std::vector<std::byte> kernel_data;
    અચલ અક્ષર *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    જો (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        જો (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } નહીંતર {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    જો (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        જો (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            પાછા ખોટું;
        }
    }

    પ્રયાસ {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        પાછા સાચું;

    } પકડો (અચલ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        પાછા ખોટું;
    }
}

સ્થિર તાર્કિક ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           અચલ std::string &              kernel_name,
                                           શૂન્ય *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           તાર્કિક                             enable_print,
                                           તાર્કિક                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    જો (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        પાછા ખોટું;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    જો (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        જો (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            પાછા ખોટું;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        જો (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            પાછા ખોટું;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    પ્રયાસ {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(સાચું);          // Wait for completion
        k_opts.setFlushL3(ખોટું);         // No L3 flush needed
        જો (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                                  static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        જો (sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            જો (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                માટે (અચલ auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (પૂર્ણાંક) error.errorCode_);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_size, k_opts);
        જો (out_event) {
            *out_event = launch_event;
        }

        જો (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_buffer, host_trace_buf.data(),
                                        ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            અચલ auto * trace_header = reinterpret_cast<અચલ trace_buffer_std_header_t *>(host_trace_buf.data());
            અચલ trace_entry_header_t * entry = nullptr;
            જ્યારે ((entry = Trace_Decode(trace_header, entry))) {
                જો (entry->type != TRACE_TYPE_STRING) {
                    ચાલુ;
                }
                અચલ auto * str_entry = reinterpret_cast<અચલ trace_string_t *>(entry);
                printf("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        જો (sync_error_check) {
            // Already triggered. No need to retrigger
            જો (!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            જો (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                માટે (અચલ auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (પૂર્ણાંક) error.errorCode_);
                }
                abort();
            }
        }

        પાછા સાચું;
    } પકડો (અચલ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        પાછા ખોટું;
    }
}

શૂન્ય ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    જો (!uk_ctx) {
        પાછા;
    }

    uk_ctx->failed = ખોટું;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

સ્થિર તાર્કિક ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    જો (!uk_ctx || !dev_ctx) {
        પાછા ખોટું;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    જો (slot.insts.empty()) {
        પાછા સાચું;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    જો (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = સાચું;
        પાછા ખોટું;
    }

    અચલ size_t   insts_size  = slot.insts.size() * કદ(ggml_et_uberkernel_inst);
    અચલ size_t   params_size = slot.params_blob.size();
    અચલ uint64_t shire_mask  = uk_ctx->shire_mask;
    તાર્કિક           ok          = ખોટું;

    પ્રયાસ {
        જો (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = સાચું;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            પાછા ખોટું;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<અચલ std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, સાચું);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    સાચું);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(કદ(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, કદ(params), shire_mask, ખોટું, ખોટું,
                                            &launch_event);
        જો (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = સાચું;
        }
    } પકડો (અચલ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    જો (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_ગણતરી;
        auto & next          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(next, runtime);
        next.insts.clear();
        next.params_blob.clear();
    } નહીંતર {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    પાછા ok;
}

શૂન્ય ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    જો (!uk_ctx) {
        પાછા;
    }

    uk_ctx->failed = ખોટું;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

તાર્કિક ggml_et_uberkernel_failed(અચલ ggml_backend_et_uberkernel_context * uk_ctx) {
    પાછા uk_ctx && uk_ctx->failed;
}

સ્થિર તાર્કિક ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      અચલ std::string &              kernel_name,
                                      શૂન્ય *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      તાર્કિક                             enable_print,
                                      તાર્કિક                             sync_error_check) {
    જો (!dev_ctx) {
        પાછા ખોટું;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    અચલ uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    જો (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        જો (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            પાછા ખોટું;
        }
        પાછા ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    અચલ size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    જો (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    અચલ std::byte * params_bytes = reinterpret_cast<અચલ std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    slot.insts.push_back(inst);

    જો (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    પાછા સાચું;
}

તાર્કિક ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    જો (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        પાછા સાચું;
    }

    પાછા ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

તાર્કિક ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           અચલ std::string &              kernel_name,
                           શૂન્ય *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           તાર્કિક                             enable_print,
                           તાર્કિક                             sync_error_check) {
    જો (!dev_ctx) {
        પાછા ખોટું;
    }

    જો (!dev_ctx->uberkernel_enabled) {
        પાછા ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    પાછા ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

શૂન્ય ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, અચલ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    જો (!runtime) {
        પાછા;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    જો (kernel_it != dev_ctx->loaded_kernels.end()) {
        પ્રયાસ {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } પકડો (અચલ std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

શૂન્ય ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    જો (!dev_ctx) {
        પાછા;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    માટે (અચલ auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    માટે (અચલ auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    માટે (અચલ auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    પાછા loaded_kernels;
}

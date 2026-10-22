#അന്തര്ഭാവ "ggml-et-kernels.h"

#അന്തര്ഭാവ "ggml-et-kernels-embed.hpp"
#അന്തര്ഭാവ "ggml-et-uberkernel-kernel-map.h"
#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <fstream>

#define ET_TRACE_DECODER_IMPL
#അന്തര്ഭാവ <et-trace/decoder.h>
#അന്തര്ഭാവ <et-trace/layout.h>

സ്ഥിതം constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

സ്ഥിതം size_t ggml_et_align_up(size_t value, size_t alignment) {
    തിരികെ (value + alignment - 1) & ~(alignment - 1);
}

സ്ഥിതം size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    എങ്കിൽ (current_capacity == 0) {
        തിരികെ required_capacity;
    }

    size_t next_capacity = current_capacity;
    ആകുമ്പോൾ (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    തിരികെ next_capacity;
}

സ്ഥിതം ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    തിരികെ uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
സ്ഥിതം ശൂന്യം ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         സ്ഥിരം std::shared_ptr<rt::IRuntime> & runtime) {
    എങ്കിൽ (!slot.has_pending || !runtime) {
        തിരികെ;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = അസത്യം;
}

സ്ഥിതം ശൂന്യം ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    എങ്കിൽ (!uk_ctx) {
        തിരികെ;
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

സ്ഥിതം തർക്കം ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_size,
                                                    size_t                            params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    എങ്കിൽ (!dev_ctx || !runtime) {
        തിരികെ അസത്യം;
    }

    ശ്രമിക്കുക {
        എങ്കിൽ (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            സ്ഥിരം size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            എങ്കിൽ (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        എങ്കിൽ (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            സ്ഥിരം size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            എങ്കിൽ (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } പിടിക്കുക (സ്ഥിരം std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        തിരികെ അസത്യം;
    }

    തിരികെ slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
സ്ഥിതം std::vector<std::byte> ggml_et_get_embedded_kernel(സ്ഥിരം std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    എങ്കിൽ (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        തിരികെ {};
    }

    സ്ഥിരം unsigned അക്ഷരം * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    തിരികെ buffer;
}

// Read kernel from file (for development/override)
സ്ഥിതം std::vector<std::byte> ggml_et_read_kernel_file(സ്ഥിരം std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    എങ്കിൽ (!file) {
        തിരികെ {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.പഠ(reinterpret_cast<അക്ഷരം *>(buffer.data()), size);

    തിരികെ buffer;
}

// Load kernel from file or embedded data
തർക്കം ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, സ്ഥിരം std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    എങ്കിൽ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        തിരികെ അസത്യം;
    }

    // Check if kernel already loaded
    എങ്കിൽ (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        തിരികെ സത്യം;
    }

    std::vector<std::byte> kernel_data;
    സ്ഥിരം അക്ഷരം *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    എങ്കിൽ (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        എങ്കിൽ (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } അല്ലെങ്കിൽ {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    എങ്കിൽ (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        എങ്കിൽ (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            തിരികെ അസത്യം;
        }
    }

    ശ്രമിക്കുക {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        തിരികെ സത്യം;

    } പിടിക്കുക (സ്ഥിരം std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        തിരികെ അസത്യം;
    }
}

സ്ഥിതം തർക്കം ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           സ്ഥിരം std::string &              kernel_name,
                                           ശൂന്യം *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           തർക്കം                             enable_print,
                                           തർക്കം                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    എങ്കിൽ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        തിരികെ അസത്യം;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    എങ്കിൽ (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        എങ്കിൽ (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            തിരികെ അസത്യം;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        എങ്കിൽ (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            തിരികെ അസത്യം;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    ശ്രമിക്കുക {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(സത്യം);          // Wait for completion
        k_opts.setFlushL3(അസത്യം);         // No L3 flush needed
        എങ്കിൽ (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                                  static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        എങ്കിൽ (sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            എങ്കിൽ (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                വേണ്ടി (സ്ഥിരം auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (പൂർണ്ണസംഖ്യ) error.errorCode_);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_size, k_opts);
        എങ്കിൽ (out_event) {
            *out_event = launch_event;
        }

        എങ്കിൽ (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_buffer, host_trace_buf.data(),
                                        ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            സ്ഥിരം auto * trace_header = reinterpret_cast<സ്ഥിരം trace_buffer_std_header_t *>(host_trace_buf.data());
            സ്ഥിരം trace_entry_header_t * entry = nullptr;
            ആകുമ്പോൾ ((entry = Trace_Decode(trace_header, entry))) {
                എങ്കിൽ (entry->type != TRACE_TYPE_STRING) {
                    തുടരുക;
                }
                സ്ഥിരം auto * str_entry = reinterpret_cast<സ്ഥിരം trace_string_t *>(entry);
                printf("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        എങ്കിൽ (sync_error_check) {
            // Already triggered. No need to retrigger
            എങ്കിൽ (!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            എങ്കിൽ (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                വേണ്ടി (സ്ഥിരം auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (പൂർണ്ണസംഖ്യ) error.errorCode_);
                }
                abort();
            }
        }

        തിരികെ സത്യം;
    } പിടിക്കുക (സ്ഥിരം std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        തിരികെ അസത്യം;
    }
}

ശൂന്യം ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    എങ്കിൽ (!uk_ctx) {
        തിരികെ;
    }

    uk_ctx->failed = അസത്യം;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

സ്ഥിതം തർക്കം ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    എങ്കിൽ (!uk_ctx || !dev_ctx) {
        തിരികെ അസത്യം;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    എങ്കിൽ (slot.insts.empty()) {
        തിരികെ സത്യം;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    എങ്കിൽ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = സത്യം;
        തിരികെ അസത്യം;
    }

    സ്ഥിരം size_t   insts_size  = slot.insts.size() * വലുപ്പം(ggml_et_uberkernel_inst);
    സ്ഥിരം size_t   params_size = slot.params_blob.size();
    സ്ഥിരം uint64_t shire_mask  = uk_ctx->shire_mask;
    തർക്കം           ok          = അസത്യം;

    ശ്രമിക്കുക {
        എങ്കിൽ (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = സത്യം;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            തിരികെ അസത്യം;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<സ്ഥിരം std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, സത്യം);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    സത്യം);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(വലുപ്പം(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, വലുപ്പം(params), shire_mask, അസത്യം, അസത്യം,
                                            &launch_event);
        എങ്കിൽ (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = സത്യം;
        }
    } പിടിക്കുക (സ്ഥിരം std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    എങ്കിൽ (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_എണ്ണം;
        auto & next          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(next, runtime);
        next.insts.clear();
        next.params_blob.clear();
    } അല്ലെങ്കിൽ {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    തിരികെ ok;
}

ശൂന്യം ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    എങ്കിൽ (!uk_ctx) {
        തിരികെ;
    }

    uk_ctx->failed = അസത്യം;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

തർക്കം ggml_et_uberkernel_failed(സ്ഥിരം ggml_backend_et_uberkernel_context * uk_ctx) {
    തിരികെ uk_ctx && uk_ctx->failed;
}

സ്ഥിതം തർക്കം ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      സ്ഥിരം std::string &              kernel_name,
                                      ശൂന്യം *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      തർക്കം                             enable_print,
                                      തർക്കം                             sync_error_check) {
    എങ്കിൽ (!dev_ctx) {
        തിരികെ അസത്യം;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    സ്ഥിരം uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    എങ്കിൽ (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        എങ്കിൽ (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            തിരികെ അസത്യം;
        }
        തിരികെ ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    സ്ഥിരം size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    എങ്കിൽ (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    സ്ഥിരം std::byte * params_bytes = reinterpret_cast<സ്ഥിരം std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    slot.insts.push_back(inst);

    എങ്കിൽ (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    തിരികെ സത്യം;
}

തർക്കം ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    എങ്കിൽ (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        തിരികെ സത്യം;
    }

    തിരികെ ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

തർക്കം ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           സ്ഥിരം std::string &              kernel_name,
                           ശൂന്യം *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           തർക്കം                             enable_print,
                           തർക്കം                             sync_error_check) {
    എങ്കിൽ (!dev_ctx) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!dev_ctx->uberkernel_enabled) {
        തിരികെ ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    തിരികെ ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

ശൂന്യം ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, സ്ഥിരം std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    എങ്കിൽ (!runtime) {
        തിരികെ;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    എങ്കിൽ (kernel_it != dev_ctx->loaded_kernels.end()) {
        ശ്രമിക്കുക {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } പിടിക്കുക (സ്ഥിരം std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

ശൂന്യം ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    എങ്കിൽ (!dev_ctx) {
        തിരികെ;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    വേണ്ടി (സ്ഥിരം auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    വേണ്ടി (സ്ഥിരം auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    വേണ്ടി (സ്ഥിരം auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    തിരികെ loaded_kernels;
}

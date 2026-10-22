#অন্তর্ভাব "ggml-et-kernels.h"

#অন্তর্ভাব "ggml-et-kernels-embed.hpp"
#অন্তর্ভাব "ggml-et-uberkernel-kernel-map.h"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <fstream>

#define ET_TRACE_DECODER_IMPL
#অন্তর্ভাব <et-trace/decoder.h>
#অন্তর্ভাব <et-trace/layout.h>

স্থির constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

স্থির size_t ggml_et_align_up(size_t value, size_t alignment) {
    প্রত্যাগম (value + alignment - 1) & ~(alignment - 1);
}

স্থির size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    যদি (current_capacity == 0) {
        প্রত্যাগম required_capacity;
    }

    size_t next_capacity = current_capacity;
    যাবত্ (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    প্রত্যাগম next_capacity;
}

স্থির ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    প্রত্যাগম uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
স্থির শূন্য ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         ধ্রুব std::shared_ptr<rt::IRuntime> & runtime) {
    যদি (!slot.has_pending || !runtime) {
        প্রত্যাগম;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = অসত্য;
}

স্থির শূন্য ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        প্রত্যাগম;
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

স্থির তার্কিক ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_size,
                                                    size_t                            params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!dev_ctx || !runtime) {
        প্রত্যাগম অসত্য;
    }

    যত্ন {
        যদি (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            ধ্রুব size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            যদি (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        যদি (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            ধ্রুব size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            যদি (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } গৃহাণ (ধ্রুব std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        প্রত্যাগম অসত্য;
    }

    প্রত্যাগম slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
স্থির std::vector<std::byte> ggml_et_get_embedded_kernel(ধ্রুব std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    যদি (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        প্রত্যাগম {};
    }

    ধ্রুব unsigned বর্ণ * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    প্রত্যাগম buffer;
}

// Read kernel from file (for development/override)
স্থির std::vector<std::byte> ggml_et_read_kernel_file(ধ্রুব std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    যদি (!file) {
        প্রত্যাগম {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.পঠ(reinterpret_cast<বর্ণ *>(buffer.data()), size);

    প্রত্যাগম buffer;
}

// Load kernel from file or embedded data
তার্কিক ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, ধ্রুব std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        প্রত্যাগম অসত্য;
    }

    // Check if kernel already loaded
    যদি (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        প্রত্যাগম সত্য;
    }

    std::vector<std::byte> kernel_data;
    ধ্রুব বর্ণ *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    যদি (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        যদি (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } অন্যথা {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    যদি (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        যদি (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            প্রত্যাগম অসত্য;
        }
    }

    যত্ন {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        প্রত্যাগম সত্য;

    } গৃহাণ (ধ্রুব std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        প্রত্যাগম অসত্য;
    }
}

স্থির তার্কিক ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           ধ্রুব std::string &              kernel_name,
                                           শূন্য *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           তার্কিক                             enable_print,
                                           তার্কিক                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        প্রত্যাগম অসত্য;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    যদি (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        যদি (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            প্রত্যাগম অসত্য;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        যদি (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            প্রত্যাগম অসত্য;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    যত্ন {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(সত্য);          // Wait for completion
        k_opts.setFlushL3(অসত্য);         // No L3 flush needed
        যদি (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                                  static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        যদি (sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            যদি (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                প্রতি (ধ্রুব auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (পূর্ণাঙ্ক) error.errorCode_);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_size, k_opts);
        যদি (out_event) {
            *out_event = launch_event;
        }

        যদি (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_buffer, host_trace_buf.data(),
                                        ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            ধ্রুব auto * trace_header = reinterpret_cast<ধ্রুব trace_buffer_std_header_t *>(host_trace_buf.data());
            ধ্রুব trace_entry_header_t * entry = nullptr;
            যাবত্ ((entry = Trace_Decode(trace_header, entry))) {
                যদি (entry->type != TRACE_TYPE_STRING) {
                    অনুবর্ত;
                }
                ধ্রুব auto * str_entry = reinterpret_cast<ধ্রুব trace_string_t *>(entry);
                printf("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        যদি (sync_error_check) {
            // Already triggered. No need to retrigger
            যদি (!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            যদি (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                প্রতি (ধ্রুব auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (পূর্ণাঙ্ক) error.errorCode_);
                }
                abort();
            }
        }

        প্রত্যাগম সত্য;
    } গৃহাণ (ধ্রুব std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        প্রত্যাগম অসত্য;
    }
}

শূন্য ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        প্রত্যাগম;
    }

    uk_ctx->failed = অসত্য;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

স্থির তার্কিক ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx || !dev_ctx) {
        প্রত্যাগম অসত্য;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    যদি (slot.insts.empty()) {
        প্রত্যাগম সত্য;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = সত্য;
        প্রত্যাগম অসত্য;
    }

    ধ্রুব size_t   insts_size  = slot.insts.size() * পরিমাণ(ggml_et_uberkernel_inst);
    ধ্রুব size_t   params_size = slot.params_blob.size();
    ধ্রুব uint64_t shire_mask  = uk_ctx->shire_mask;
    তার্কিক           ok          = অসত্য;

    যত্ন {
        যদি (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = সত্য;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            প্রত্যাগম অসত্য;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<ধ্রুব std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, সত্য);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    সত্য);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(পরিমাণ(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, পরিমাণ(params), shire_mask, অসত্য, অসত্য,
                                            &launch_event);
        যদি (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = সত্য;
        }
    } গৃহাণ (ধ্রুব std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    যদি (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_গুনতি;
        auto & next          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(next, runtime);
        next.insts.clear();
        next.params_blob.clear();
    } অন্যথা {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    প্রত্যাগম ok;
}

শূন্য ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        প্রত্যাগম;
    }

    uk_ctx->failed = অসত্য;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

তার্কিক ggml_et_uberkernel_failed(ধ্রুব ggml_backend_et_uberkernel_context * uk_ctx) {
    প্রত্যাগম uk_ctx && uk_ctx->failed;
}

স্থির তার্কিক ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ধ্রুব std::string &              kernel_name,
                                      শূন্য *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      তার্কিক                             enable_print,
                                      তার্কিক                             sync_error_check) {
    যদি (!dev_ctx) {
        প্রত্যাগম অসত্য;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ধ্রুব uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    যদি (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        যদি (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            প্রত্যাগম অসত্য;
        }
        প্রত্যাগম ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    ধ্রুব size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    যদি (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    ধ্রুব std::byte * params_bytes = reinterpret_cast<ধ্রুব std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    slot.insts.push_back(inst);

    যদি (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    প্রত্যাগম সত্য;
}

তার্কিক ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    যদি (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        প্রত্যাগম সত্য;
    }

    প্রত্যাগম ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

তার্কিক ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ধ্রুব std::string &              kernel_name,
                           শূন্য *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           তার্কিক                             enable_print,
                           তার্কিক                             sync_error_check) {
    যদি (!dev_ctx) {
        প্রত্যাগম অসত্য;
    }

    যদি (!dev_ctx->uberkernel_enabled) {
        প্রত্যাগম ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    প্রত্যাগম ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

শূন্য ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ধ্রুব std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        প্রত্যাগম;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    যদি (kernel_it != dev_ctx->loaded_kernels.end()) {
        যত্ন {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } গৃহাণ (ধ্রুব std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

শূন্য ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    যদি (!dev_ctx) {
        প্রত্যাগম;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    প্রতি (ধ্রুব auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    প্রতি (ধ্রুব auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    প্রতি (ধ্রুব auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    প্রত্যাগম loaded_kernels;
}

#অন্তর্ভাৱ "ggml-et-kernels.h"

#অন্তর্ভাৱ "ggml-et-kernels-embed.hpp"
#অন্তর্ভাৱ "ggml-et-uberkernel-kernel-map.h"
#অন্তর্ভাৱ "ggml-impl.h"

#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <fstream>

#define ET_TRACE_DECODER_IMPL
#অন্তর্ভাৱ <et-trace/decoder.h>
#অন্তর্ভাৱ <et-trace/layout.h>

স্থির constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

স্থির size_t ggml_et_align_up(size_t value, size_t alignment) {
    ঘূৰাই (value + alignment - 1) & ~(alignment - 1);
}

স্থির size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    যদি (current_capacity == 0) {
        ঘূৰাই required_capacity;
    }

    size_t next_capacity = current_capacity;
    যেতিয়ালৈ (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    ঘূৰাই next_capacity;
}

স্থির ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    ঘূৰাই uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
স্থির শূন্য ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         ধ্রুৱ std::shared_ptr<rt::IRuntime> & runtime) {
    যদি (!slot.has_pending || !runtime) {
        ঘূৰাই;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = মিছা;
}

স্থির শূন্য ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        ঘূৰাই;
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
        ঘূৰাই মিছা;
    }

    যত্ন {
        যদি (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            ধ্রুৱ size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            যদি (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        যদি (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            ধ্রুৱ size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            যদি (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } গৃহাণ (ধ্রুৱ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        ঘূৰাই মিছা;
    }

    ঘূৰাই slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
স্থির std::vector<std::byte> ggml_et_get_embedded_kernel(ধ্রুৱ std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    যদি (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        ঘূৰাই {};
    }

    ধ্রুৱ unsigned ৱর্ণ * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    ঘূৰাই buffer;
}

// Read kernel from file (for development/override)
স্থির std::vector<std::byte> ggml_et_read_kernel_file(ধ্রুৱ std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    যদি (!file) {
        ঘূৰাই {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.পঠ(reinterpret_cast<ৱর্ণ *>(buffer.data()), size);

    ঘূৰাই buffer;
}

// Load kernel from file or embedded data
তার্কিক ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, ধ্রুৱ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        ঘূৰাই মিছা;
    }

    // Check if kernel already loaded
    যদি (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        ঘূৰাই সঁচা;
    }

    std::vector<std::byte> kernel_data;
    ধ্রুৱ ৱর্ণ *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

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
            ঘূৰাই মিছা;
        }
    }

    যত্ন {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        ঘূৰাই সঁচা;

    } গৃহাণ (ধ্রুৱ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        ঘূৰাই মিছা;
    }
}

স্থির তার্কিক ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           ধ্রুৱ std::string &              kernel_name,
                                           শূন্য *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           তার্কিক                             enable_print,
                                           তার্কিক                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        ঘূৰাই মিছা;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    যদি (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        যদি (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            ঘূৰাই মিছা;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        যদি (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            ঘূৰাই মিছা;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    যত্ন {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(সঁচা);          // Wait for completion
        k_opts.setFlushL3(মিছা);         // No L3 flush needed
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
                বাবে (ধ্রুৱ auto & error : errors) {
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
            ধ্রুৱ auto * trace_header = reinterpret_cast<ধ্রুৱ trace_buffer_std_header_t *>(host_trace_buf.data());
            ধ্রুৱ trace_entry_header_t * entry = nullptr;
            যেতিয়ালৈ ((entry = Trace_Decode(trace_header, entry))) {
                যদি (entry->type != TRACE_TYPE_STRING) {
                    অনুৱর্ত;
                }
                ধ্রুৱ auto * str_entry = reinterpret_cast<ধ্রুৱ trace_string_t *>(entry);
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
                বাবে (ধ্রুৱ auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (পূর্ণাঙ্ক) error.errorCode_);
                }
                abort();
            }
        }

        ঘূৰাই সঁচা;
    } গৃহাণ (ধ্রুৱ std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        ঘূৰাই মিছা;
    }
}

শূন্য ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        ঘূৰাই;
    }

    uk_ctx->failed = মিছা;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

স্থির তার্কিক ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx || !dev_ctx) {
        ঘূৰাই মিছা;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    যদি (slot.insts.empty()) {
        ঘূৰাই সঁচা;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = সঁচা;
        ঘূৰাই মিছা;
    }

    ধ্রুৱ size_t   insts_size  = slot.insts.size() * পরিমাণ(ggml_et_uberkernel_inst);
    ধ্রুৱ size_t   params_size = slot.params_blob.size();
    ধ্রুৱ uint64_t shire_mask  = uk_ctx->shire_mask;
    তার্কিক           ok          = মিছা;

    যত্ন {
        যদি (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = সঁচা;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            ঘূৰাই মিছা;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<ধ্রুৱ std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, সঁচা);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    সঁচা);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(পরিমাণ(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, পরিমাণ(params), shire_mask, মিছা, মিছা,
                                            &launch_event);
        যদি (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = সঁচা;
        }
    } গৃহাণ (ধ্রুৱ std::exception & e) {
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
    ঘূৰাই ok;
}

শূন্য ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        ঘূৰাই;
    }

    uk_ctx->failed = মিছা;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

তার্কিক ggml_et_uberkernel_failed(ধ্রুৱ ggml_backend_et_uberkernel_context * uk_ctx) {
    ঘূৰাই uk_ctx && uk_ctx->failed;
}

স্থির তার্কিক ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ধ্রুৱ std::string &              kernel_name,
                                      শূন্য *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      তার্কিক                             enable_print,
                                      তার্কিক                             sync_error_check) {
    যদি (!dev_ctx) {
        ঘূৰাই মিছা;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ধ্রুৱ uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    যদি (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        যদি (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            ঘূৰাই মিছা;
        }
        ঘূৰাই ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    ধ্রুৱ size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    যদি (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    ধ্রুৱ std::byte * params_bytes = reinterpret_cast<ধ্রুৱ std::byte *>(params);
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

    ঘূৰাই সঁচা;
}

তার্কিক ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    যদি (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        ঘূৰাই সঁচা;
    }

    ঘূৰাই ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

তার্কিক ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ধ্রুৱ std::string &              kernel_name,
                           শূন্য *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           তার্কিক                             enable_print,
                           তার্কিক                             sync_error_check) {
    যদি (!dev_ctx) {
        ঘূৰাই মিছা;
    }

    যদি (!dev_ctx->uberkernel_enabled) {
        ঘূৰাই ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    ঘূৰাই ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

শূন্য ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ধ্রুৱ std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        ঘূৰাই;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    যদি (kernel_it != dev_ctx->loaded_kernels.end()) {
        যত্ন {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } গৃহাণ (ধ্রুৱ std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

শূন্য ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    যদি (!dev_ctx) {
        ঘূৰাই;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    বাবে (ধ্রুৱ auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    বাবে (ধ্রুৱ auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    বাবে (ধ্রুৱ auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    ঘূৰাই loaded_kernels;
}

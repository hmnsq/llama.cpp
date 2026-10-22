#شامل "ggml-et-kernels.h"

#شامل "ggml-et-kernels-embed.hpp"
#شامل "ggml-et-uberkernel-kernel-map.h"
#شامل "ggml-impl.h"

#شامل <cstdlib>
#شامل <cstring>
#شامل <fstream>

#define ET_TRACE_DECODER_IMPL
#شامل <et-trace/decoder.h>
#شامل <et-trace/layout.h>

ایستا constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

ایستا size_t ggml_et_align_up(size_t value, size_t alignment) {
    بازگشت (value + alignment - 1) & ~(alignment - 1);
}

ایستا size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    اگر (current_capacity == 0) {
        بازگشت required_capacity;
    }

    size_t next_capacity = current_capacity;
    وقتی (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    بازگشت next_capacity;
}

ایستا ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    بازگشت uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
ایستا تهی ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         ثابت std::shared_ptr<rt::IRuntime> & runtime) {
    اگر (!slot.has_pending || !runtime) {
        بازگشت;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = نادرست;
}

ایستا تهی ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    اگر (!uk_ctx) {
        بازگشت;
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

ایستا منطقی ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_size,
                                                    size_t                            params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!dev_ctx || !runtime) {
        بازگشت نادرست;
    }

    تلاش {
        اگر (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            ثابت size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            اگر (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        اگر (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            ثابت size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            اگر (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } بگیر (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        بازگشت نادرست;
    }

    بازگشت slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
ایستا std::vector<std::byte> ggml_et_get_embedded_kernel(ثابت std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    اگر (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        بازگشت {};
    }

    ثابت unsigned نویسه * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    بازگشت buffer;
}

// Read kernel from file (for development/override)
ایستا std::vector<std::byte> ggml_et_read_kernel_file(ثابت std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    اگر (!file) {
        بازگشت {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.اقرأ(reinterpret_cast<نویسه *>(buffer.data()), size);

    بازگشت buffer;
}

// Load kernel from file or embedded data
منطقی ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        بازگشت نادرست;
    }

    // Check if kernel already loaded
    اگر (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        بازگشت درست;
    }

    std::vector<std::byte> kernel_data;
    ثابت نویسه *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    اگر (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        اگر (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } وگرنه {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    اگر (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        اگر (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            بازگشت نادرست;
        }
    }

    تلاش {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        بازگشت درست;

    } بگیر (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        بازگشت نادرست;
    }
}

ایستا منطقی ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           ثابت std::string &              kernel_name,
                                           تهی *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           منطقی                             enable_print,
                                           منطقی                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        بازگشت نادرست;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    اگر (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        اگر (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            بازگشت نادرست;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        اگر (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            بازگشت نادرست;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    تلاش {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(درست);          // Wait for completion
        k_opts.setFlushL3(نادرست);         // No L3 flush needed
        اگر (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                                  static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        اگر (sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            اگر (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                برای (ثابت auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحیح) error.errorCode_);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_size, k_opts);
        اگر (out_event) {
            *out_event = launch_event;
        }

        اگر (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_buffer, host_trace_buf.data(),
                                        ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            ثابت auto * trace_header = reinterpret_cast<ثابت trace_buffer_std_header_t *>(host_trace_buf.data());
            ثابت trace_entry_header_t * entry = nullptr;
            وقتی ((entry = Trace_Decode(trace_header, entry))) {
                اگر (entry->type != TRACE_TYPE_STRING) {
                    ادامه;
                }
                ثابت auto * str_entry = reinterpret_cast<ثابت trace_string_t *>(entry);
                printf("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        اگر (sync_error_check) {
            // Already triggered. No need to retrigger
            اگر (!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            اگر (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                برای (ثابت auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحیح) error.errorCode_);
                }
                abort();
            }
        }

        بازگشت درست;
    } بگیر (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        بازگشت نادرست;
    }
}

تهی ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    اگر (!uk_ctx) {
        بازگشت;
    }

    uk_ctx->failed = نادرست;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ایستا منطقی ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    اگر (!uk_ctx || !dev_ctx) {
        بازگشت نادرست;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    اگر (slot.insts.empty()) {
        بازگشت درست;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = درست;
        بازگشت نادرست;
    }

    ثابت size_t   insts_size  = slot.insts.size() * اندازه(ggml_et_uberkernel_inst);
    ثابت size_t   params_size = slot.params_blob.size();
    ثابت uint64_t shire_mask  = uk_ctx->shire_mask;
    منطقی           ok          = نادرست;

    تلاش {
        اگر (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = درست;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            بازگشت نادرست;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<ثابت std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, درست);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    درست);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(اندازه(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, اندازه(params), shire_mask, نادرست, نادرست,
                                            &launch_event);
        اگر (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = درست;
        }
    } بگیر (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    اگر (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_شمار;
        auto & next          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(next, runtime);
        next.insts.clear();
        next.params_blob.clear();
    } وگرنه {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    بازگشت ok;
}

تهی ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    اگر (!uk_ctx) {
        بازگشت;
    }

    uk_ctx->failed = نادرست;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

منطقی ggml_et_uberkernel_failed(ثابت ggml_backend_et_uberkernel_context * uk_ctx) {
    بازگشت uk_ctx && uk_ctx->failed;
}

ایستا منطقی ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ثابت std::string &              kernel_name,
                                      تهی *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      منطقی                             enable_print,
                                      منطقی                             sync_error_check) {
    اگر (!dev_ctx) {
        بازگشت نادرست;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ثابت uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    اگر (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        اگر (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            بازگشت نادرست;
        }
        بازگشت ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    ثابت size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    اگر (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    ثابت std::byte * params_bytes = reinterpret_cast<ثابت std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    slot.insts.push_back(inst);

    اگر (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    بازگشت درست;
}

منطقی ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    اگر (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        بازگشت درست;
    }

    بازگشت ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

منطقی ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ثابت std::string &              kernel_name,
                           تهی *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           منطقی                             enable_print,
                           منطقی                             sync_error_check) {
    اگر (!dev_ctx) {
        بازگشت نادرست;
    }

    اگر (!dev_ctx->uberkernel_enabled) {
        بازگشت ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    بازگشت ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

تهی ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        بازگشت;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    اگر (kernel_it != dev_ctx->loaded_kernels.end()) {
        تلاش {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } بگیر (ثابت std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

تهی ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    اگر (!dev_ctx) {
        بازگشت;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    برای (ثابت auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    برای (ثابت auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    برای (ثابت auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    بازگشت loaded_kernels;
}

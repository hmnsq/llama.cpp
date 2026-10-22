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
    ফেরত (value + alignment - 1) & ~(alignment - 1);
}

স্থির size_t ggml_et_পরবর্তী_capacity(size_t current_capacity, size_t required_capacity) {
    যদি (current_capacity == 0) {
        ফেরত required_capacity;
    }

    size_t পরবর্তী_capacity = current_capacity;
    যখন (পরবর্তী_capacity < required_capacity) {
        পরবর্তী_capacity *= 2;
    }

    ফেরত পরবর্তী_capacity;
}

স্থির ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    ফেরত uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
স্থির শূন্য ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         ধ্রুবক std::shared_ptr<rt::IRuntime> & runtime) {
    যদি (!slot.has_pending || !runtime) {
        ফেরত;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = মিথ্যা;
}

স্থির শূন্য ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        ফেরত;
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

স্থির যৌক্তিক ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_আকার,
                                                    size_t                            params_আকার) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!dev_ctx || !runtime) {
        ফেরত মিথ্যা;
    }

    চেষ্টা {
        যদি (slot.device_insts == nullptr || insts_আকার > slot.device_insts_capacity) {
            ধ্রুবক size_t new_capacity = ggml_et_পরবর্তী_capacity(slot.device_insts_capacity, insts_আকার);
            যদি (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        যদি (slot.device_params == nullptr || params_আকার > slot.device_params_capacity) {
            ধ্রুবক size_t new_capacity = ggml_et_পরবর্তী_capacity(slot.device_params_capacity, params_আকার);
            যদি (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } ধরো (ধ্রুবক std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        ফেরত মিথ্যা;
    }

    ফেরত slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
স্থির std::vector<std::byte> ggml_et_get_embedded_kernel(ধ্রুবক std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    যদি (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        ফেরত {};
    }

    ধ্রুবক unsigned অক্ষর * তথ্য = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> বাফার(size);
    std::memcpy(বাফার.তথ্য(), তথ্য, size);

    ফেরত বাফার;
}

// Read kernel from file (for development/override)
স্থির std::vector<std::byte> ggml_et_read_kernel_file(ধ্রুবক std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    যদি (!file) {
        ফেরত {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> বাফার(size);
    file.পঠ(reinterpret_cast<অক্ষর *>(বাফার.তথ্য()), size);

    ফেরত বাফার;
}

// Load kernel from file or embedded data
যৌক্তিক ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, ধ্রুবক std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        ফেরত মিথ্যা;
    }

    // Check if kernel already loaded
    যদি (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        ফেরত সত্য;
    }

    std::vector<std::byte> kernel_তথ্য;
    ধ্রুবক অক্ষর *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    যদি (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_তথ্য             = ggml_et_read_kernel_file(kernel_file);

        যদি (!kernel_তথ্য.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } অন্যথা {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    যদি (kernel_তথ্য.empty()) {
        kernel_তথ্য = ggml_et_get_embedded_kernel(kernel_name);
        যদি (kernel_তথ্য.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            ফেরত মিথ্যা;
        }
    }

    চেষ্টা {
        // Load kernel code using device's default stream
        auto load_ফলাফল = runtime->loadCode(dev_ctx->default_stream, kernel_তথ্য.তথ্য(), kernel_তথ্য.size());
        runtime->waitForEvent(load_ফলাফল.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_ফলাফল.kernel_;
        ফেরত সত্য;

    } ধরো (ধ্রুবক std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        ফেরত মিথ্যা;
    }
}

স্থির যৌক্তিক ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           ধ্রুবক std::string &              kernel_name,
                                           শূন্য *                           params,
                                           size_t                           params_আকার,
                                           uint64_t                         shire_mask,
                                           যৌক্তিক                             enable_print,
                                           যৌক্তিক                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        ফেরত মিথ্যা;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    যদি (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        যদি (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            ফেরত মিথ্যা;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        যদি (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            ফেরত মিথ্যা;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    চেষ্টা {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(সত্য);          // Wait for completion
        k_opts.setFlushL3(মিথ্যা);         // No L3 flush needed
        যদি (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_বাফার),
                                  static_cast<uint32_t>(ET_TRACE_বাফার_আকার),
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
                জন্য (ধ্রুবক auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (পূর্ণসংখ্যা) error.errorCode_);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_আকার, k_opts);
        যদি (out_event) {
            *out_event = launch_event;
        }

        যদি (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_বাফার_আকার);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_বাফার, host_trace_buf.তথ্য(),
                                        ET_TRACE_বাফার_আকার);
            runtime->waitForStream(dev_ctx->default_stream);
            ধ্রুবক auto * trace_header = reinterpret_cast<ধ্রুবক trace_বাফার_std_header_t *>(host_trace_buf.তথ্য());
            ধ্রুবক trace_entry_header_t * entry = nullptr;
            যখন ((entry = Trace_Decode(trace_header, entry))) {
                যদি (entry->type != TRACE_TYPE_STRING) {
                    চালু;
                }
                ধ্রুবক auto * str_entry = reinterpret_cast<ধ্রুবক trace_string_t *>(entry);
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
                জন্য (ধ্রুবক auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (পূর্ণসংখ্যা) error.errorCode_);
                }
                abort();
            }
        }

        ফেরত সত্য;
    } ধরো (ধ্রুবক std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        ফেরত মিথ্যা;
    }
}

শূন্য ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        ফেরত;
    }

    uk_ctx->failed = মিথ্যা;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

স্থির যৌক্তিক ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx || !dev_ctx) {
        ফেরত মিথ্যা;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    যদি (slot.insts.empty()) {
        ফেরত সত্য;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = সত্য;
        ফেরত মিথ্যা;
    }

    ধ্রুবক size_t   insts_আকার  = slot.insts.size() * আকার(ggml_et_uberkernel_inst);
    ধ্রুবক size_t   params_আকার = slot.params_blob.size();
    ধ্রুবক uint64_t shire_mask  = uk_ctx->shire_mask;
    যৌক্তিক           ok          = মিথ্যা;

    চেষ্টা {
        যদি (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_আকার, params_আকার)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = সত্য;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            ফেরত মিথ্যা;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<ধ্রুবক std::byte *>(slot.insts.তথ্য()),
                                    slot.device_insts, insts_আকার, সত্য);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.তথ্য(), slot.device_params, params_আকার,
                                    সত্য);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(আকার(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, আকার(params), shire_mask, মিথ্যা, মিথ্যা,
                                            &launch_event);
        যদি (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = সত্য;
        }
    } ধরো (ধ্রুবক std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    যদি (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_গুনতি;
        auto & পরবর্তী          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(পরবর্তী, runtime);
        পরবর্তী.insts.clear();
        পরবর্তী.params_blob.clear();
    } অন্যথা {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    ফেরত ok;
}

শূন্য ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    যদি (!uk_ctx) {
        ফেরত;
    }

    uk_ctx->failed = মিথ্যা;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

যৌক্তিক ggml_et_uberkernel_failed(ধ্রুবক ggml_backend_et_uberkernel_context * uk_ctx) {
    ফেরত uk_ctx && uk_ctx->failed;
}

স্থির যৌক্তিক ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ধ্রুবক std::string &              kernel_name,
                                      শূন্য *                           params,
                                      size_t                           params_আকার,
                                      uint64_t                         shire_mask,
                                      যৌক্তিক                             enable_print,
                                      যৌক্তিক                             sync_error_check) {
    যদি (!dev_ctx) {
        ফেরত মিথ্যা;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ধ্রুবক uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    যদি (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        যদি (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            ফেরত মিথ্যা;
        }
        ফেরত ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_আকার, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    ধ্রুবক size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    যদি (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    ধ্রুবক std::byte * params_bytes = reinterpret_cast<ধ্রুবক std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_আকার);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_আকার),
    };
    slot.insts.push_back(inst);

    যদি (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    ফেরত সত্য;
}

যৌক্তিক ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    যদি (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        ফেরত সত্য;
    }

    ফেরত ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

যৌক্তিক ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ধ্রুবক std::string &              kernel_name,
                           শূন্য *                           params,
                           size_t                           params_আকার,
                           uint64_t                         shire_mask,
                           যৌক্তিক                             enable_print,
                           যৌক্তিক                             sync_error_check) {
    যদি (!dev_ctx) {
        ফেরত মিথ্যা;
    }

    যদি (!dev_ctx->uberkernel_enabled) {
        ফেরত ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_আকার, shire_mask, enable_print,
                                              sync_error_check);
    }

    ফেরত ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_আকার, shire_mask, enable_print,
                                     sync_error_check);
}

শূন্য ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ধ্রুবক std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    যদি (!runtime) {
        ফেরত;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    যদি (kernel_it != dev_ctx->loaded_kernels.end()) {
        চেষ্টা {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } ধরো (ধ্রুবক std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

শূন্য ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    যদি (!dev_ctx) {
        ফেরত;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    জন্য (ধ্রুবক auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    জন্য (ধ্রুবক auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    জন্য (ধ্রুবক auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    ফেরত loaded_kernels;
}

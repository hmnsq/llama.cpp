#समावेश "ggml-et-kernels.h"

#समावेश "ggml-et-kernels-embed.hpp"
#समावेश "ggml-et-uberkernel-kernel-map.h"
#समावेश "ggml-impl.h"

#समावेश <cstdlib>
#समावेश <cstring>
#समावेश <fstream>

#घोषणा ET_TRACE_DECODER_IMPL
#समावेश <et-trace/decoder.h>
#समावेश <et-trace/layout.h>

अटल constexpr माप_प्रकार GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

अटल माप_प्रकार ggml_et_align_up(माप_प्रकार value, माप_प्रकार alignment) {
    वापस (value + alignment - 1) & ~(alignment - 1);
}

अटल माप_प्रकार ggml_et_अग्र_capacity(माप_प्रकार current_capacity, माप_प्रकार required_capacity) {
    अगर (current_capacity == 0) {
        वापस required_capacity;
    }

    माप_प्रकार अग्र_capacity = current_capacity;
    जबतक (अग्र_capacity < required_capacity) {
        अग्र_capacity *= 2;
    }

    वापस अग्र_capacity;
}

अटल ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    वापस uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
अटल व्योम ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         स्थिर std::shared_ptr<rt::IRuntime> & runtime) {
    अगर (!slot.has_pending || !runtime) {
        वापस;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = false;
}

अटल व्योम ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    अगर (!uk_ctx) {
        वापस;
    }

    uk_ctx->shire_mask = 0;
    स्वतः & slot        = ggml_et_uberkernel_current_slot(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_slot_wait(slot, ggml_et_runtime());
    slot.insts.सफा();
    slot.params_blob.सफा();
}

अटल bool ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    माप_प्रकार                            insts_आकार,
                                                    माप_प्रकार                            params_आकार) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!dev_ctx || !runtime) {
        वापस false;
    }

    प्रयास {
        अगर (slot.device_insts == nullptr || insts_आकार > slot.device_insts_capacity) {
            स्थिर माप_प्रकार new_capacity = ggml_et_अग्र_capacity(slot.device_insts_capacity, insts_आकार);
            अगर (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        अगर (slot.device_params == nullptr || params_आकार > slot.device_params_capacity) {
            स्थिर माप_प्रकार new_capacity = ggml_et_अग्र_capacity(slot.device_params_capacity, params_आकार);
            अगर (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } पकड़ो (स्थिर std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        वापस false;
    }

    वापस slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
अटल std::vector<std::byte> ggml_et_get_embedded_kernel(स्थिर std::string & kernel_name) {
    स्वतः it = ggml_et_embedded_kernels.find(kernel_name);
    अगर (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        वापस {};
    }

    स्थिर अचिन्हित अक्षर * तथ्य = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> बफर(size);
    std::स_नकल(बफर.तथ्य(), तथ्य, size);

    वापस बफर;
}

// Read kernel from file (for development/override)
अटल std::vector<std::byte> ggml_et_read_kernel_file(स्थिर std::string & kernel_path) {
    std::अख_प्रवाह file(kernel_path, std::पनप::द्विग | std::पनप::अंत_पर);
    अगर (!file) {
        वापस {};
    }

    स्वतः size = file.tellg();
    file.ग_स्थान(0, std::पनप::beg);

    std::vector<std::byte> बफर(size);
    file.पढ़ो(reinterpret_cast<अक्षर *>(बफर.तथ्य()), size);

    वापस बफर;
}

// Load kernel from file or embedded data
bool ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, स्थिर std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        वापस false;
    }

    // Check if kernel already loaded
    अगर (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        वापस true;
    }

    std::vector<std::byte> kernel_तथ्य;
    स्थिर अक्षर *           kernels_path = दो_पर्या("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    अगर (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_तथ्य             = ggml_et_read_kernel_file(kernel_file);

        अगर (!kernel_तथ्य.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } अन्यथा {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    अगर (kernel_तथ्य.empty()) {
        kernel_तथ्य = ggml_et_get_embedded_kernel(kernel_name);
        अगर (kernel_तथ्य.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            वापस false;
        }
    }

    प्रयास {
        // Load kernel code using device's default stream
        स्वतः load_परिणाम = runtime->loadCode(dev_ctx->default_stream, kernel_तथ्य.तथ्य(), kernel_तथ्य.size());
        runtime->waitForEvent(load_परिणाम.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_परिणाम.kernel_;
        वापस true;

    } पकड़ो (स्थिर std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        वापस false;
    }
}

अटल bool ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           स्थिर std::string &              kernel_name,
                                           व्योम *                           params,
                                           माप_प्रकार                           params_आकार,
                                           uint64_t                         shire_mask,
                                           bool                             enable_print,
                                           bool                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        वापस false;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    स्वतः kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    अगर (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        अगर (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            वापस false;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        अगर (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            वापस false;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    प्रयास {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(true);          // Wait for completion
        k_opts.setFlushL3(false);         // No L3 flush needed
        अगर (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_बफर),
                                  static_cast<uint32_t>(ET_TRACE_बफर_आकार),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        अगर (sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            स्वतः errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            अगर (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                क्रम (स्थिर स्वतः & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (पूर्णांक) error.errorCode_);
                }
                पात();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_आकार, k_opts);
        अगर (out_event) {
            *out_event = launch_event;
        }

        अगर (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_बफर_आकार);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_बफर, host_trace_buf.तथ्य(),
                                        ET_TRACE_बफर_आकार);
            runtime->waitForStream(dev_ctx->default_stream);
            स्थिर स्वतः * trace_header = reinterpret_cast<स्थिर trace_बफर_std_header_t *>(host_trace_buf.तथ्य());
            स्थिर trace_entry_header_t * entry = nullptr;
            जबतक ((entry = Trace_Decode(trace_header, entry))) {
                अगर (entry->type != TRACE_TYPE_STRING) {
                    जारी;
                }
                स्थिर स्वतः * str_entry = reinterpret_cast<स्थिर trace_string_t *>(entry);
                म_लिखो("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        अगर (sync_error_check) {
            // Already triggered. No need to retrigger
            अगर (!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            स्वतः errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            अगर (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                क्रम (स्थिर स्वतः & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (पूर्णांक) error.errorCode_);
                }
                पात();
            }
        }

        वापस true;
    } पकड़ो (स्थिर std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        वापस false;
    }
}

व्योम ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    अगर (!uk_ctx) {
        वापस;
    }

    uk_ctx->failed = false;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

अटल bool ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    अगर (!uk_ctx || !dev_ctx) {
        वापस false;
    }

    स्वतः & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    अगर (slot.insts.empty()) {
        वापस true;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = true;
        वापस false;
    }

    स्थिर माप_प्रकार   insts_आकार  = slot.insts.size() * माप(ggml_et_uberkernel_inst);
    स्थिर माप_प्रकार   params_आकार = slot.params_blob.size();
    स्थिर uint64_t shire_mask  = uk_ctx->shire_mask;
    bool           ok          = false;

    प्रयास {
        अगर (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_आकार, params_आकार)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = true;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.सफा();
            slot.params_blob.सफा();
            uk_ctx->shire_mask = 0;
            वापस false;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<स्थिर std::byte *>(slot.insts.तथ्य()),
                                    slot.device_insts, insts_आकार, true);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.तथ्य(), slot.device_params, params_आकार,
                                    true);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(माप(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, माप(params), shire_mask, false, false,
                                            &launch_event);
        अगर (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = true;
        }
    } पकड़ो (स्थिर std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    अगर (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_गिनती;
        स्वतः & अग्र          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(अग्र, runtime);
        अग्र.insts.सफा();
        अग्र.params_blob.सफा();
    } अन्यथा {
        slot.insts.सफा();
        slot.params_blob.सफा();
    }
    uk_ctx->shire_mask = 0;
    वापस ok;
}

व्योम ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    अगर (!uk_ctx) {
        वापस;
    }

    uk_ctx->failed = false;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

bool ggml_et_uberkernel_failed(स्थिर ggml_backend_et_uberkernel_context * uk_ctx) {
    वापस uk_ctx && uk_ctx->failed;
}

अटल bool ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      स्थिर std::string &              kernel_name,
                                      व्योम *                           params,
                                      माप_प्रकार                           params_आकार,
                                      uint64_t                         shire_mask,
                                      bool                             enable_print,
                                      bool                             sync_error_check) {
    अगर (!dev_ctx) {
        वापस false;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    स्थिर uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    अगर (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        अगर (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            वापस false;
        }
        वापस ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_आकार, shire_mask, enable_print,
                                              sync_error_check);
    }

    स्वतः &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    स्थिर माप_प्रकार params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    अगर (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    स्थिर std::byte * params_bytes = reinterpret_cast<स्थिर std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_आकार);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_आकार),
    };
    slot.insts.push_back(inst);

    अगर (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    वापस true;
}

bool ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    अगर (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        वापस true;
    }

    वापस ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

bool ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           स्थिर std::string &              kernel_name,
                           व्योम *                           params,
                           माप_प्रकार                           params_आकार,
                           uint64_t                         shire_mask,
                           bool                             enable_print,
                           bool                             sync_error_check) {
    अगर (!dev_ctx) {
        वापस false;
    }

    अगर (!dev_ctx->uberkernel_enabled) {
        वापस ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_आकार, shire_mask, enable_print,
                                              sync_error_check);
    }

    वापस ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_आकार, shire_mask, enable_print,
                                     sync_error_check);
}

व्योम ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, स्थिर std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    अगर (!runtime) {
        वापस;
    }

    स्वतः kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    अगर (kernel_it != dev_ctx->loaded_kernels.end()) {
        प्रयास {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } पकड़ो (स्थिर std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

व्योम ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    अगर (!dev_ctx) {
        वापस;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    क्रम (स्थिर स्वतः & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    क्रम (स्थिर स्वतः & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    क्रम (स्थिर स्वतः & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    वापस loaded_kernels;
}

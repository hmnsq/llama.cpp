#अन्तर्भाव "ggml-et-kernels.h"

#अन्तर्भाव "ggml-et-kernels-embed.hpp"
#अन्तर्भाव "ggml-et-uberkernel-kernel-map.h"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <fstream>

#परिभाषय ET_TRACE_DECODER_IMPL
#अन्तर्भाव <et-trace/decoder.h>
#अन्तर्भाव <et-trace/layout.h>

स्थिर constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

स्थिर size_t ggml_et_align_ऊर्ध्वम्(size_t value, size_t alignment) {
    प्रत्यागम (value + alignment - 1) & ~(alignment - 1);
}

स्थिर size_t ggml_et_अग्र_क्षमता(size_t current_क्षमता, size_t required_क्षमता) {
    यदि (current_क्षमता == 0) {
        प्रत्यागम required_क्षमता;
    }

    size_t अग्र_क्षमता = current_क्षमता;
    यावत् (अग्र_क्षमता < required_क्षमता) {
        अग्र_क्षमता *= 2;
    }

    प्रत्यागम अग्र_क्षमता;
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

    uk_ctx->shire_आवरणम् = 0;
    स्वचालित & slot        = ggml_et_uberkernel_current_slot(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_slot_wait(slot, ggml_et_runtime());
    slot.insts.clear();
    slot.params_blob.clear();
}

स्थिर तार्किक ggml_et_uberkernel_ensure_slot_क्षमता(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_आकार,
                                                    size_t                            params_आकार) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!dev_ctx || !runtime) {
        प्रत्यागम असत्य;
    }

    यत्न {
        यदि (slot.device_insts == शून्यसूचकः || insts_आकार > slot.device_insts_क्षमता) {
            ध्रुव size_t new_क्षमता = ggml_et_अग्र_क्षमता(slot.device_insts_क्षमता, insts_आकार);
            यदि (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_क्षमता);
            slot.device_insts_क्षमता = slot.device_insts ? new_क्षमता : 0;
        }

        यदि (slot.device_params == शून्यसूचकः || params_आकार > slot.device_params_क्षमता) {
            ध्रुव size_t new_क्षमता = ggml_et_अग्र_क्षमता(slot.device_params_क्षमता, params_आकार);
            यदि (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_क्षमता);
            slot.device_params_क्षमता = slot.device_params ? new_क्षमता : 0;
        }
    } गृहाण (ध्रुव std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        प्रत्यागम असत्य;
    }

    प्रत्यागम slot.device_insts != शून्यसूचकः && slot.device_params != शून्यसूचकः;
}

// Get embedded kernel data by name
स्थिर std::vector<std::byte> ggml_et_get_embedded_kernel(ध्रुव std::string & kernel_name) {
    स्वचालित it = ggml_et_embedded_kernels.find(kernel_name);
    यदि (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        प्रत्यागम {};
    }

    ध्रुव अचिह्नित वर्ण * दत्त = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> बफर(size);
    std::स्मृतिं_प्रतिलिख(बफर.दत्त(), दत्त, size);

    प्रत्यागम बफर;
}

// Read kernel from file (for development/override)
स्थिर std::vector<std::byte> ggml_et_read_kernel_file(ध्रुव std::string & kernel_मार्गः) {
    std::ifstream file(kernel_मार्गः, std::ios::binary | std::ios::ate);
    यदि (!file) {
        प्रत्यागम {};
    }

    स्वचालित size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> बफर(size);
    file.पठ(reinterpret_cast<वर्ण *>(बफर.दत्त()), size);

    प्रत्यागम बफर;
}

// Load kernel from file or embedded data
तार्किक ggml_et_आरोपय_kernel(ggml_backend_et_device_context * dev_ctx, ध्रुव std::string & kernel_name) {
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

    std::vector<std::byte> kernel_दत्त;
    ध्रुव वर्ण *           kernels_मार्गः = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    यदि (kernels_मार्गः) {
        std::string kernel_file = std::string(kernels_मार्गः) + "/" + kernel_name + ".elf";
        kernel_दत्त             = ggml_et_read_kernel_file(kernel_file);

        यदि (!kernel_दत्त.रिक्तानि()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } अन्यथा {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    यदि (kernel_दत्त.रिक्तानि()) {
        kernel_दत्त = ggml_et_get_embedded_kernel(kernel_name);
        यदि (kernel_दत्त.रिक्तानि()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            प्रत्यागम असत्य;
        }
    }

    यत्न {
        // Load kernel code using device's default stream
        स्वचालित आरोपय_परिणाम = runtime->आरोपय_संहिता(dev_ctx->default_प्रवाहः, kernel_दत्त.दत्त(), kernel_दत्त.size());
        runtime->waitForEvent(आरोपय_परिणाम.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = आरोपय_परिणाम.kernel_;
        प्रत्यागम सत्य;

    } गृहाण (ध्रुव std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        प्रत्यागम असत्य;
    }
}

स्थिर तार्किक ggml_et_launch_kernel_आन्तरिकम्(ggml_backend_et_device_context * dev_ctx,
                                           ध्रुव std::string &              kernel_name,
                                           शून्य *                           params,
                                           size_t                           params_आकार,
                                           uint64_t                         shire_आवरणम्,
                                           तार्किक                             enable_print,
                                           तार्किक                             sync_error_check,
                                           rt::EventId *                    बहिर्गमनम्_event = शून्यसूचकः) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        प्रत्यागम असत्य;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    स्वचालित kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    यदि (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        यदि (!ggml_et_आरोपय_kernel(dev_ctx, kernel_name)) {
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
        k_opts.समुच्चयः_Shire_आवरणम्(shire_आवरणम्);  // Default: all shires (0xFFFFFFFF)
        k_opts.समुच्चयः_Barrier(सत्य);          // Wait for completion
        k_opts.समुच्चयः_Flush_L3(असत्य);         // No L3 flush needed
        यदि (enable_print) {
            k_opts.समुच्चयः_User_Tracing(reinterpret_cast<uint64_t>(dev_ctx->trace_बफर),
                                  static_cast<uint32_t>(ET_TRACE_बफर_आकार),
                                  0,                      //देहली
                                  shire_आवरणम्,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        यदि (sync_error_check) {
            runtime->wait_For_प्रवाहः(dev_ctx->default_प्रवाहः);
            स्वचालित errors = runtime->retrieve_प्रवाहः_Errors(dev_ctx->default_प्रवाहः);
            यदि (!errors.रिक्तानि()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                प्रति (ध्रुव स्वचालित & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (पूर्णाङ्क) error.error_संहिता);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_प्रवाहः, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_आकार, k_opts);
        यदि (बहिर्गमनम्_event) {
            *बहिर्गमनम्_event = launch_event;
        }

        यदि (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_बफर_आकार);
            runtime->memcpy_Device_गन्तव्यम्_Host(dev_ctx->default_प्रवाहः, dev_ctx->trace_बफर, host_trace_buf.दत्त(),
                                        ET_TRACE_बफर_आकार);
            runtime->wait_For_प्रवाहः(dev_ctx->default_प्रवाहः);
            ध्रुव स्वचालित * trace_शीर्षकम् = reinterpret_cast<ध्रुव trace_बफर_std_शीर्षकम्_t *>(host_trace_buf.दत्त());
            ध्रुव trace_प्रविष्टिः_शीर्षकम्_t * प्रविष्टिः = शून्यसूचकः;
            यावत् ((प्रविष्टिः = Trace_Decode(trace_शीर्षकम्, प्रविष्टिः))) {
                यदि (प्रविष्टिः->type != TRACE_TYPE_STRING) {
                    अनुवर्त;
                }
                ध्रुव स्वचालित * str_प्रविष्टिः = reinterpret_cast<ध्रुव trace_string_t *>(प्रविष्टिः);
                प्रारूप_लिख("[hart %d] %s", प्रविष्टिः->hart_id, str_प्रविष्टिः->string);
            }
        }

        यदि (sync_error_check) {
            // Already triggered. No need to retrigger
            यदि (!enable_print) {
                runtime->wait_For_प्रवाहः(dev_ctx->default_प्रवाहः);
            }
            स्वचालित errors = runtime->retrieve_प्रवाहः_Errors(dev_ctx->default_प्रवाहः);
            यदि (!errors.रिक्तानि()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                प्रति (ध्रुव स्वचालित & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (पूर्णाङ्क) error.error_संहिता);
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

    स्वचालित & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    यदि (slot.insts.रिक्तानि()) {
        प्रत्यागम सत्य;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = सत्य;
        प्रत्यागम असत्य;
    }

    ध्रुव size_t   insts_आकार  = slot.insts.size() * परिमाण(ggml_et_uberkernel_inst);
    ध्रुव size_t   params_आकार = slot.params_blob.size();
    ध्रुव uint64_t shire_आवरणम्  = uk_ctx->shire_आवरणम्;
    तार्किक           ok          = असत्य;

    यत्न {
        यदि (!ggml_et_uberkernel_ensure_slot_क्षमता(slot, dev_ctx, insts_आकार, params_आकार)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = सत्य;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_आवरणम् = 0;
            प्रत्यागम असत्य;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpy_Host_गन्तव्यम्_Device(dev_ctx->default_प्रवाहः, reinterpret_cast<ध्रुव std::byte *>(slot.insts.दत्त()),
                                    slot.device_insts, insts_आकार, सत्य);
        runtime->memcpy_Host_गन्तव्यम्_Device(dev_ctx->default_प्रवाहः, slot.params_blob.दत्त(), slot.device_params, params_आकार,
                                    सत्य);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(परिमाण(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_आन्तरिकम्(dev_ctx, "uberkernel", &params, परिमाण(params), shire_आवरणम्, असत्य, असत्य,
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
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_गणनम्;
        स्वचालित & अग्र          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(अग्र, runtime);
        अग्र.insts.clear();
        अग्र.params_blob.clear();
    } अन्यथा {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_आवरणम् = 0;
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
                                      size_t                           params_आकार,
                                      uint64_t                         shire_आवरणम्,
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
        प्रत्यागम ggml_et_launch_kernel_आन्तरिकम्(dev_ctx, kernel_name, params, params_आकार, shire_आवरणम्, enable_print,
                                              sync_error_check);
    }

    स्वचालित &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    ध्रुव size_t params_विस्थापनम् = ggml_et_align_ऊर्ध्वम्(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    यदि (params_विस्थापनम् > slot.params_blob.size()) {
        slot.params_blob.resize(params_विस्थापनम्);
    }

    ध्रुव std::byte * params_अष्टकानि = reinterpret_cast<ध्रुव std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_अष्टकानि, params_अष्टकानि + params_आकार);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_विस्थापनम्),
        static_cast<uint32_t>(params_आकार),
    };
    slot.insts.push_back(inst);

    यदि (slot.insts.size() == 1) {
        uk_ctx->shire_आवरणम् = shire_आवरणम्;
    }

    प्रत्यागम सत्य;
}

तार्किक ggml_et_uberkernel_अन्तः_graph(ggml_backend_et_device_context * dev_ctx) {
    यदि (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        प्रत्यागम सत्य;
    }

    प्रत्यागम ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

तार्किक ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ध्रुव std::string &              kernel_name,
                           शून्य *                           params,
                           size_t                           params_आकार,
                           uint64_t                         shire_आवरणम्,
                           तार्किक                             enable_print,
                           तार्किक                             sync_error_check) {
    यदि (!dev_ctx) {
        प्रत्यागम असत्य;
    }

    यदि (!dev_ctx->uberkernel_enabled) {
        प्रत्यागम ggml_et_launch_kernel_आन्तरिकम्(dev_ctx, kernel_name, params, params_आकार, shire_आवरणम्, enable_print,
                                              sync_error_check);
    }

    प्रत्यागम ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_आकार, shire_आवरणम्, enable_print,
                                     sync_error_check);
}

शून्य ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ध्रुव std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        प्रत्यागम;
    }

    स्वचालित kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    यदि (kernel_it != dev_ctx->loaded_kernels.end()) {
        यत्न {
            runtime->unload_संहिता(kernel_it->second);
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
    std::vector<std::string> kernel_नामानि;
    kernel_नामानि.reserve(dev_ctx->loaded_kernels.size());
    प्रति (ध्रुव स्वचालित & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_नामानि.push_back(kernel_pair.first);
    }

    प्रति (ध्रुव स्वचालित & kernel_name : kernel_नामानि) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    प्रति (ध्रुव स्वचालित & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    प्रत्यागम loaded_kernels;
}

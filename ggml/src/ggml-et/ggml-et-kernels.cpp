#అన్తర్భావ "ggml-et-kernels.h"

#అన్తర్భావ "ggml-et-kernels-embed.hpp"
#అన్తర్భావ "ggml-et-uberkernel-kernel-map.h"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <fstream>

#నిర్వచించు ET_TRACE_DECODER_IMPL
#అన్తర్భావ <et-trace/decoder.h>
#అన్తర్భావ <et-trace/layout.h>

స్థిర constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

స్థిర size_t ggml_et_align_పైకి(size_t value, size_t alignment) {
    తిరిగి (value + alignment - 1) & ~(alignment - 1);
}

స్థిర size_t ggml_et_తదుపరి_సామర్థ్యం(size_t current_సామర్థ్యం, size_t required_సామర్థ్యం) {
    ఒకవేళ (current_సామర్థ్యం == 0) {
        తిరిగి required_సామర్థ్యం;
    }

    size_t తదుపరి_సామర్థ్యం = current_సామర్థ్యం;
    అయితే (తదుపరి_సామర్థ్యం < required_సామర్థ్యం) {
        తదుపరి_సామర్థ్యం *= 2;
    }

    తిరిగి తదుపరి_సామర్థ్యం;
}

స్థిర ggml_backend_et_uberkernel_స్థానం & ggml_et_uberkernel_current_స్థానం(ggml_backend_et_uberkernel_context * uk_ctx) {
    తిరిగి uk_ctx->slots[uk_ctx->current_స్థానం];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
స్థిర శూన్య ggml_et_uberkernel_స్థానం_wait(ggml_backend_et_uberkernel_స్థానం &     స్థానం,
                                         స్థిరరాశి std::shared_ptr<rt::IRuntime> & runtime) {
    ఒకవేళ (!స్థానం.has_pending || !runtime) {
        తిరిగి;
    }
    runtime->waitForEvent(స్థానం.pending_event);
    స్థానం.has_pending = అబద్ధం;
}

స్థిర శూన్య ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    ఒకవేళ (!uk_ctx) {
        తిరిగి;
    }

    uk_ctx->shire_ముసుగు = 0;
    స్వయంచాలక & స్థానం        = ggml_et_uberkernel_current_స్థానం(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_స్థానం_wait(స్థానం, ggml_et_runtime());
    స్థానం.insts.clear();
    స్థానం.params_blob.clear();
}

స్థిర తార్కిక ggml_et_uberkernel_ensure_స్థానం_సామర్థ్యం(ggml_backend_et_uberkernel_స్థానం & స్థానం,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_కొలత,
                                                    size_t                            params_కొలత) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!dev_ctx || !runtime) {
        తిరిగి అబద్ధం;
    }

    ప్రయత్నించు {
        ఒకవేళ (స్థానం.device_insts == శూన్యసూచిక || insts_కొలత > స్థానం.device_insts_సామర్థ్యం) {
            స్థిరరాశి size_t new_సామర్థ్యం = ggml_et_తదుపరి_సామర్థ్యం(స్థానం.device_insts_సామర్థ్యం, insts_కొలత);
            ఒకవేళ (స్థానం.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, స్థానం.device_insts);
            }
            స్థానం.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_సామర్థ్యం);
            స్థానం.device_insts_సామర్థ్యం = స్థానం.device_insts ? new_సామర్థ్యం : 0;
        }

        ఒకవేళ (స్థానం.device_params == శూన్యసూచిక || params_కొలత > స్థానం.device_params_సామర్థ్యం) {
            స్థిరరాశి size_t new_సామర్థ్యం = ggml_et_తదుపరి_సామర్థ్యం(స్థానం.device_params_సామర్థ్యం, params_కొలత);
            ఒకవేళ (స్థానం.device_params) {
                runtime->freeDevice(dev_ctx->rtid, స్థానం.device_params);
            }
            స్థానం.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_సామర్థ్యం);
            స్థానం.device_params_సామర్థ్యం = స్థానం.device_params ? new_సామర్థ్యం : 0;
        }
    } పట్టుకో (స్థిరరాశి std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        తిరిగి అబద్ధం;
    }

    తిరిగి స్థానం.device_insts != శూన్యసూచిక && స్థానం.device_params != శూన్యసూచిక;
}

// Get embedded kernel data by name
స్థిర std::vector<std::byte> ggml_et_get_embedded_kernel(స్థిరరాశి std::string & kernel_name) {
    స్వయంచాలక it = ggml_et_embedded_kernels.find(kernel_name);
    ఒకవేళ (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        తిరిగి {};
    }

    స్థిరరాశి గుర్తులేని అక్షరం * దత్తాంశం = it->second.మొదటి;
    uint64_t              size = it->second.second;

    std::vector<std::byte> బఫరు(size);
    std::జ్ఞాపకం_నకలు(బఫరు.దత్తాంశం(), దత్తాంశం, size);

    తిరిగి బఫరు;
}

// Read kernel from file (for development/override)
స్థిర std::vector<std::byte> ggml_et_read_kernel_file(స్థిరరాశి std::string & kernel_మార్గం) {
    std::ifstream file(kernel_మార్గం, std::ios::binary | std::ios::ate);
    ఒకవేళ (!file) {
        తిరిగి {};
    }

    స్వయంచాలక size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> బఫరు(size);
    file.పఠ(reinterpret_cast<అక్షరం *>(బఫరు.దత్తాంశం()), size);

    తిరిగి బఫరు;
}

// Load kernel from file or embedded data
తార్కిక ggml_et_ఎక్కించు_kernel(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        తిరిగి అబద్ధం;
    }

    // Check if kernel already loaded
    ఒకవేళ (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        తిరిగి నిజం;
    }

    std::vector<std::byte> kernel_దత్తాంశం;
    స్థిరరాశి అక్షరం *           kernels_మార్గం = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    ఒకవేళ (kernels_మార్గం) {
        std::string kernel_file = std::string(kernels_మార్గం) + "/" + kernel_name + ".elf";
        kernel_దత్తాంశం             = ggml_et_read_kernel_file(kernel_file);

        ఒకవేళ (!kernel_దత్తాంశం.ఖాళీలు()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } లేకపోతే {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    ఒకవేళ (kernel_దత్తాంశం.ఖాళీలు()) {
        kernel_దత్తాంశం = ggml_et_get_embedded_kernel(kernel_name);
        ఒకవేళ (kernel_దత్తాంశం.ఖాళీలు()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            తిరిగి అబద్ధం;
        }
    }

    ప్రయత్నించు {
        // Load kernel code using device's default stream
        స్వయంచాలక ఎక్కించు_ఫలితం = runtime->ఎక్కించు_సంకేతం(dev_ctx->default_ప్రవాహం, kernel_దత్తాంశం.దత్తాంశం(), kernel_దత్తాంశం.size());
        runtime->waitForEvent(ఎక్కించు_ఫలితం.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = ఎక్కించు_ఫలితం.kernel_;
        తిరిగి నిజం;

    } పట్టుకో (స్థిరరాశి std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        తిరిగి అబద్ధం;
    }
}

స్థిర తార్కిక ggml_et_launch_kernel_అంతర్గత(ggml_backend_et_device_context * dev_ctx,
                                           స్థిరరాశి std::string &              kernel_name,
                                           శూన్య *                           params,
                                           size_t                           params_కొలత,
                                           uint64_t                         shire_ముసుగు,
                                           తార్కిక                             enable_print,
                                           తార్కిక                             sync_error_check,
                                           rt::EventId *                    బహిర్గతం_event = శూన్యసూచిక) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        తిరిగి అబద్ధం;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    స్వయంచాలక kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ఒకవేళ (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        ఒకవేళ (!ggml_et_ఎక్కించు_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            తిరిగి అబద్ధం;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        ఒకవేళ (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            తిరిగి అబద్ధం;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    ప్రయత్నించు {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.సమితి_Shire_ముసుగు(shire_ముసుగు);  // Default: all shires (0xFFFFFFFF)
        k_opts.సమితి_Barrier(నిజం);          // Wait for completion
        k_opts.సమితి_Flush_L3(అబద్ధం);         // No L3 flush needed
        ఒకవేళ (enable_print) {
            k_opts.సమితి_User_Tracing(reinterpret_cast<uint64_t>(dev_ctx->trace_బఫరు),
                                  static_cast<uint32_t>(ET_TRACE_బఫరు_కొలత),
                                  0,                      //పరిమితి
                                  shire_ముసుగు,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        ఒకవేళ (sync_error_check) {
            runtime->wait_For_ప్రవాహం(dev_ctx->default_ప్రవాహం);
            స్వయంచాలక errors = runtime->retrieve_ప్రవాహం_Errors(dev_ctx->default_ప్రవాహం);
            ఒకవేళ (!errors.ఖాళీలు()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                కొరకు (స్థిరరాశి స్వయంచాలక & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (పూర్ణాంకం) error.error_సంకేతం);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_ప్రవాహం, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_కొలత, k_opts);
        ఒకవేళ (బహిర్గతం_event) {
            *బహిర్గతం_event = launch_event;
        }

        ఒకవేళ (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_బఫరు_కొలత);
            runtime->memcpy_Device_గమ్యం_Host(dev_ctx->default_ప్రవాహం, dev_ctx->trace_బఫరు, host_trace_buf.దత్తాంశం(),
                                        ET_TRACE_బఫరు_కొలత);
            runtime->wait_For_ప్రవాహం(dev_ctx->default_ప్రవాహం);
            స్థిరరాశి స్వయంచాలక * trace_శీర్షిక = reinterpret_cast<స్థిరరాశి trace_బఫరు_std_శీర్షిక_t *>(host_trace_buf.దత్తాంశం());
            స్థిరరాశి trace_నమోదు_శీర్షిక_t * నమోదు_2 = శూన్యసూచిక;
            అయితే ((నమోదు_2 = Trace_Decode(trace_శీర్షిక, నమోదు_2))) {
                ఒకవేళ (నమోదు_2->type != TRACE_TYPE_STRING) {
                    కొనసాగు;
                }
                స్థిరరాశి స్వయంచాలక * str_నమోదు = reinterpret_cast<స్థిరరాశి trace_string_t *>(నమోదు_2);
                రాయి("[hart %d] %s", నమోదు_2->hart_id, str_నమోదు->string);
            }
        }

        ఒకవేళ (sync_error_check) {
            // Already triggered. No need to retrigger
            ఒకవేళ (!enable_print) {
                runtime->wait_For_ప్రవాహం(dev_ctx->default_ప్రవాహం);
            }
            స్వయంచాలక errors = runtime->retrieve_ప్రవాహం_Errors(dev_ctx->default_ప్రవాహం);
            ఒకవేళ (!errors.ఖాళీలు()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                కొరకు (స్థిరరాశి స్వయంచాలక & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (పూర్ణాంకం) error.error_సంకేతం);
                }
                abort();
            }
        }

        తిరిగి నిజం;
    } పట్టుకో (స్థిరరాశి std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        తిరిగి అబద్ధం;
    }
}

శూన్య ggml_et_uberkernel_begin_లేఖాచిత్రం(ggml_backend_et_uberkernel_context * uk_ctx) {
    ఒకవేళ (!uk_ctx) {
        తిరిగి;
    }

    uk_ctx->failed = అబద్ధం;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

స్థిర తార్కిక ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    ఒకవేళ (!uk_ctx || !dev_ctx) {
        తిరిగి అబద్ధం;
    }

    స్వయంచాలక & స్థానం = ggml_et_uberkernel_current_స్థానం(uk_ctx);
    ఒకవేళ (స్థానం.insts.ఖాళీలు()) {
        తిరిగి నిజం;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = నిజం;
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి size_t   insts_కొలత  = స్థానం.insts.size() * పరిమాణం(ggml_et_uberkernel_inst);
    స్థిరరాశి size_t   params_కొలత = స్థానం.params_blob.size();
    స్థిరరాశి uint64_t shire_ముసుగు  = uk_ctx->shire_ముసుగు;
    తార్కిక           ok          = అబద్ధం;

    ప్రయత్నించు {
        ఒకవేళ (!ggml_et_uberkernel_ensure_స్థానం_సామర్థ్యం(స్థానం, dev_ctx, insts_కొలత, params_కొలత)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = నిజం;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            స్థానం.insts.clear();
            స్థానం.params_blob.clear();
            uk_ctx->shire_ముసుగు = 0;
            తిరిగి అబద్ధం;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpy_Host_గమ్యం_Device(dev_ctx->default_ప్రవాహం, reinterpret_cast<స్థిరరాశి std::byte *>(స్థానం.insts.దత్తాంశం()),
                                    స్థానం.device_insts, insts_కొలత, నిజం);
        runtime->memcpy_Host_గమ్యం_Device(dev_ctx->default_ప్రవాహం, స్థానం.params_blob.దత్తాంశం(), స్థానం.device_params, params_కొలత,
                                    నిజం);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(స్థానం.insts.size()),
            static_cast<uint32_t>(పరిమాణం(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(స్థానం.device_insts),
            reinterpret_cast<uint64_t>(స్థానం.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_అంతర్గత(dev_ctx, "uberkernel", &params, పరిమాణం(params), shire_ముసుగు, అబద్ధం, అబద్ధం,
                                            &launch_event);
        ఒకవేళ (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            స్థానం.pending_event = launch_event;
            స్థానం.has_pending   = నిజం;
        }
    } పట్టుకో (స్థిరరాశి std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    ఒకవేళ (ok) {
        uk_ctx->current_స్థానం = (uk_ctx->current_స్థానం + 1) % ggml_backend_et_uberkernel_context::స్థానం_లెక్క_2;
        స్వయంచాలక & తదుపరి          = ggml_et_uberkernel_current_స్థానం(uk_ctx);
        ggml_et_uberkernel_స్థానం_wait(తదుపరి, runtime);
        తదుపరి.insts.clear();
        తదుపరి.params_blob.clear();
    } లేకపోతే {
        స్థానం.insts.clear();
        స్థానం.params_blob.clear();
    }
    uk_ctx->shire_ముసుగు = 0;
    తిరిగి ok;
}

శూన్య ggml_et_uberkernel_abort_లేఖాచిత్రం(ggml_backend_et_uberkernel_context * uk_ctx) {
    ఒకవేళ (!uk_ctx) {
        తిరిగి;
    }

    uk_ctx->failed = అబద్ధం;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

తార్కిక ggml_et_uberkernel_failed(స్థిరరాశి ggml_backend_et_uberkernel_context * uk_ctx) {
    తిరిగి uk_ctx && uk_ctx->failed;
}

స్థిర తార్కిక ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      స్థిరరాశి std::string &              kernel_name,
                                      శూన్య *                           params,
                                      size_t                           params_కొలత,
                                      uint64_t                         shire_ముసుగు,
                                      తార్కిక                             enable_print,
                                      తార్కిక                             sync_error_check) {
    ఒకవేళ (!dev_ctx) {
        తిరిగి అబద్ధం;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    స్థిరరాశి uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    ఒకవేళ (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        ఒకవేళ (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            తిరిగి అబద్ధం;
        }
        తిరిగి ggml_et_launch_kernel_అంతర్గత(dev_ctx, kernel_name, params, params_కొలత, shire_ముసుగు, enable_print,
                                              sync_error_check);
    }

    స్వయంచాలక &       స్థానం          = ggml_et_uberkernel_current_స్థానం(uk_ctx);
    స్థిరరాశి size_t params_స్థానభ్రంశం = ggml_et_align_పైకి(స్థానం.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    ఒకవేళ (params_స్థానభ్రంశం > స్థానం.params_blob.size()) {
        స్థానం.params_blob.resize(params_స్థానభ్రంశం);
    }

    స్థిరరాశి std::byte * params_బైట్లు = reinterpret_cast<స్థిరరాశి std::byte *>(params);
    స్థానం.params_blob.insert(స్థానం.params_blob.end(), params_బైట్లు, params_బైట్లు + params_కొలత);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_స్థానభ్రంశం),
        static_cast<uint32_t>(params_కొలత),
    };
    స్థానం.insts.push_back(inst);

    ఒకవేళ (స్థానం.insts.size() == 1) {
        uk_ctx->shire_ముసుగు = shire_ముసుగు;
    }

    తిరిగి నిజం;
}

తార్కిక ggml_et_uberkernel_చివర_లేఖాచిత్రం(ggml_backend_et_device_context * dev_ctx) {
    ఒకవేళ (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        తిరిగి నిజం;
    }

    తిరిగి ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

తార్కిక ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           స్థిరరాశి std::string &              kernel_name,
                           శూన్య *                           params,
                           size_t                           params_కొలత,
                           uint64_t                         shire_ముసుగు,
                           తార్కిక                             enable_print,
                           తార్కిక                             sync_error_check) {
    ఒకవేళ (!dev_ctx) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!dev_ctx->uberkernel_enabled) {
        తిరిగి ggml_et_launch_kernel_అంతర్గత(dev_ctx, kernel_name, params, params_కొలత, shire_ముసుగు, enable_print,
                                              sync_error_check);
    }

    తిరిగి ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_కొలత, shire_ముసుగు, enable_print,
                                     sync_error_check);
}

శూన్య ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, స్థిరరాశి std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        తిరిగి;
    }

    స్వయంచాలక kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ఒకవేళ (kernel_it != dev_ctx->loaded_kernels.end()) {
        ప్రయత్నించు {
            runtime->unload_సంకేతం(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } పట్టుకో (స్థిరరాశి std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

శూన్య ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    ఒకవేళ (!dev_ctx) {
        తిరిగి;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_పేర్లు;
    kernel_పేర్లు.reserve(dev_ctx->loaded_kernels.size());
    కొరకు (స్థిరరాశి స్వయంచాలక & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_పేర్లు.push_back(kernel_pair.మొదటి);
    }

    కొరకు (స్థిరరాశి స్వయంచాలక & kernel_name : kernel_పేర్లు) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    కొరకు (స్థిరరాశి స్వయంచాలక & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    తిరిగి loaded_kernels;
}

#شامل "ggml-et-kernels.h"

#شامل "ggml-et-kernels-embed.hpp"
#شامل "ggml-et-uberkernel-kernel-map.h"
#شامل "ggml-impl.h"

#شامل <cstdlib>
#شامل <cstring>
#شامل <fstream>

#تعریف_کرو ET_TRACE_DECODER_IMPL
#شامل <et-trace/decoder.h>
#شامل <et-trace/layout.h>

ساکن constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

ساکن size_t ggml_et_align_اوپر(size_t value, size_t alignment) {
    واپس (value + alignment - 1) & ~(alignment - 1);
}

ساکن size_t ggml_et_اگلا_استعداد(size_t current_استعداد, size_t required_استعداد) {
    اگر (current_استعداد == 0) {
        واپس required_استعداد;
    }

    size_t اگلا_استعداد = current_استعداد;
    جب (اگلا_استعداد < required_استعداد) {
        اگلا_استعداد *= 2;
    }

    واپس اگلا_استعداد;
}

ساکن ggml_backend_et_uberkernel_خانہ & ggml_et_uberkernel_current_خانہ(ggml_backend_et_uberkernel_context * uk_ctx) {
    واپس uk_ctx->slots[uk_ctx->current_خانہ];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
ساکن خالی ggml_et_uberkernel_خانہ_wait(ggml_backend_et_uberkernel_خانہ &     خانہ,
                                         ثابت std::shared_ptr<rt::IRuntime> & runtime) {
    اگر (!خانہ.has_pending || !runtime) {
        واپس;
    }
    runtime->waitForEvent(خانہ.pending_event);
    خانہ.has_pending = جھوٹ;
}

ساکن خالی ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    اگر (!uk_ctx) {
        واپس;
    }

    uk_ctx->shire_نقاب = 0;
    خودکار & خانہ        = ggml_et_uberkernel_current_خانہ(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_خانہ_wait(خانہ, ggml_et_runtime());
    خانہ.insts.clear();
    خانہ.params_blob.clear();
}

ساکن منطقی ggml_et_uberkernel_ensure_خانہ_استعداد(ggml_backend_et_uberkernel_خانہ & خانہ,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_سائز,
                                                    size_t                            params_سائز) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!dev_ctx || !runtime) {
        واپس جھوٹ;
    }

    کوشش {
        اگر (خانہ.device_insts == خالی_اشاریہ || insts_سائز > خانہ.device_insts_استعداد) {
            ثابت size_t new_استعداد = ggml_et_اگلا_استعداد(خانہ.device_insts_استعداد, insts_سائز);
            اگر (خانہ.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, خانہ.device_insts);
            }
            خانہ.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_استعداد);
            خانہ.device_insts_استعداد = خانہ.device_insts ? new_استعداد : 0;
        }

        اگر (خانہ.device_params == خالی_اشاریہ || params_سائز > خانہ.device_params_استعداد) {
            ثابت size_t new_استعداد = ggml_et_اگلا_استعداد(خانہ.device_params_استعداد, params_سائز);
            اگر (خانہ.device_params) {
                runtime->freeDevice(dev_ctx->rtid, خانہ.device_params);
            }
            خانہ.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_استعداد);
            خانہ.device_params_استعداد = خانہ.device_params ? new_استعداد : 0;
        }
    } پکڑو (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        واپس جھوٹ;
    }

    واپس خانہ.device_insts != خالی_اشاریہ && خانہ.device_params != خالی_اشاریہ;
}

// Get embedded kernel data by name
ساکن std::vector<std::byte> ggml_et_get_embedded_kernel(ثابت std::string & kernel_name) {
    خودکار it = ggml_et_embedded_kernels.find(kernel_name);
    اگر (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        واپس {};
    }

    ثابت غیرعلامتی حرف * ڈیٹا = it->second.پہلا;
    uint64_t              size = it->second.second;

    std::vector<std::byte> بفر(size);
    std::حافظہ_نقل(بفر.ڈیٹا(), ڈیٹا, size);

    واپس بفر;
}

// Read kernel from file (for development/override)
ساکن std::vector<std::byte> ggml_et_read_kernel_file(ثابت std::string & kernel_راستہ) {
    std::ifstream file(kernel_راستہ, std::ios::binary | std::ios::ate);
    اگر (!file) {
        واپس {};
    }

    خودکار size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> بفر(size);
    file.اقرأ(reinterpret_cast<حرف *>(بفر.ڈیٹا()), size);

    واپس بفر;
}

// Load kernel from file or embedded data
منطقی ggml_et_بھرو_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        واپس جھوٹ;
    }

    // Check if kernel already loaded
    اگر (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        واپس سچ;
    }

    std::vector<std::byte> kernel_ڈیٹا;
    ثابت حرف *           kernels_راستہ = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    اگر (kernels_راستہ) {
        std::string kernel_file = std::string(kernels_راستہ) + "/" + kernel_name + ".elf";
        kernel_ڈیٹا             = ggml_et_read_kernel_file(kernel_file);

        اگر (!kernel_ڈیٹا.خالی_خانے()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } ورنہ {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    اگر (kernel_ڈیٹا.خالی_خانے()) {
        kernel_ڈیٹا = ggml_et_get_embedded_kernel(kernel_name);
        اگر (kernel_ڈیٹا.خالی_خانے()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            واپس جھوٹ;
        }
    }

    کوشش {
        // Load kernel code using device's default stream
        خودکار بھرو_نتیجہ = runtime->بھرو_رمز(dev_ctx->default_دھارا, kernel_ڈیٹا.ڈیٹا(), kernel_ڈیٹا.size());
        runtime->waitForEvent(بھرو_نتیجہ.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = بھرو_نتیجہ.kernel_;
        واپس سچ;

    } پکڑو (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        واپس جھوٹ;
    }
}

ساکن منطقی ggml_et_launch_kernel_اندرونی(ggml_backend_et_device_context * dev_ctx,
                                           ثابت std::string &              kernel_name,
                                           خالی *                           params,
                                           size_t                           params_سائز,
                                           uint64_t                         shire_نقاب,
                                           منطقی                             enable_print,
                                           منطقی                             sync_error_check,
                                           rt::EventId *                    برآمد_event = خالی_اشاریہ) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        واپس جھوٹ;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    خودکار kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    اگر (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        اگر (!ggml_et_بھرو_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            واپس جھوٹ;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        اگر (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            واپس جھوٹ;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    کوشش {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.سیٹ_Shire_نقاب(shire_نقاب);  // Default: all shires (0xFFFFFFFF)
        k_opts.سیٹ_Barrier(سچ);          // Wait for completion
        k_opts.سیٹ_Flush_L3(جھوٹ);         // No L3 flush needed
        اگر (enable_print) {
            k_opts.سیٹ_User_Tracing(reinterpret_cast<uint64_t>(dev_ctx->trace_بفر),
                                  static_cast<uint32_t>(ET_TRACE_بفر_سائز),
                                  0,                      //حد
                                  shire_نقاب,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        اگر (sync_error_check) {
            runtime->wait_For_دھارا(dev_ctx->default_دھارا);
            خودکار errors = runtime->retrieve_دھارا_Errors(dev_ctx->default_دھارا);
            اگر (!errors.خالی_خانے()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                برائے (ثابت خودکار & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحیح) error.error_رمز);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_دھارا, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_سائز, k_opts);
        اگر (برآمد_event) {
            *برآمد_event = launch_event;
        }

        اگر (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_بفر_سائز);
            runtime->memcpy_Device_مقصد_Host(dev_ctx->default_دھارا, dev_ctx->trace_بفر, host_trace_buf.ڈیٹا(),
                                        ET_TRACE_بفر_سائز);
            runtime->wait_For_دھارا(dev_ctx->default_دھارا);
            ثابت خودکار * trace_سرخی = reinterpret_cast<ثابت trace_بفر_std_سرخی_t *>(host_trace_buf.ڈیٹا());
            ثابت trace_اندراج_سرخی_t * اندراج_2 = خالی_اشاریہ;
            جب ((اندراج_2 = Trace_Decode(trace_سرخی, اندراج_2))) {
                اگر (اندراج_2->type != TRACE_TYPE_STRING) {
                    جاری;
                }
                ثابت خودکار * str_اندراج = reinterpret_cast<ثابت trace_string_t *>(اندراج_2);
                لکھو("[hart %d] %s", اندراج_2->hart_id, str_اندراج->string);
            }
        }

        اگر (sync_error_check) {
            // Already triggered. No need to retrigger
            اگر (!enable_print) {
                runtime->wait_For_دھارا(dev_ctx->default_دھارا);
            }
            خودکار errors = runtime->retrieve_دھارا_Errors(dev_ctx->default_دھارا);
            اگر (!errors.خالی_خانے()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                برائے (ثابت خودکار & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحیح) error.error_رمز);
                }
                abort();
            }
        }

        واپس سچ;
    } پکڑو (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        واپس جھوٹ;
    }
}

خالی ggml_et_uberkernel_begin_گراف(ggml_backend_et_uberkernel_context * uk_ctx) {
    اگر (!uk_ctx) {
        واپس;
    }

    uk_ctx->failed = جھوٹ;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ساکن منطقی ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    اگر (!uk_ctx || !dev_ctx) {
        واپس جھوٹ;
    }

    خودکار & خانہ = ggml_et_uberkernel_current_خانہ(uk_ctx);
    اگر (خانہ.insts.خالی_خانے()) {
        واپس سچ;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = سچ;
        واپس جھوٹ;
    }

    ثابت size_t   insts_سائز  = خانہ.insts.size() * حجم(ggml_et_uberkernel_inst);
    ثابت size_t   params_سائز = خانہ.params_blob.size();
    ثابت uint64_t shire_نقاب  = uk_ctx->shire_نقاب;
    منطقی           ok          = جھوٹ;

    کوشش {
        اگر (!ggml_et_uberkernel_ensure_خانہ_استعداد(خانہ, dev_ctx, insts_سائز, params_سائز)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = سچ;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            خانہ.insts.clear();
            خانہ.params_blob.clear();
            uk_ctx->shire_نقاب = 0;
            واپس جھوٹ;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpy_Host_مقصد_Device(dev_ctx->default_دھارا, reinterpret_cast<ثابت std::byte *>(خانہ.insts.ڈیٹا()),
                                    خانہ.device_insts, insts_سائز, سچ);
        runtime->memcpy_Host_مقصد_Device(dev_ctx->default_دھارا, خانہ.params_blob.ڈیٹا(), خانہ.device_params, params_سائز,
                                    سچ);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(خانہ.insts.size()),
            static_cast<uint32_t>(حجم(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(خانہ.device_insts),
            reinterpret_cast<uint64_t>(خانہ.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_اندرونی(dev_ctx, "uberkernel", &params, حجم(params), shire_نقاب, جھوٹ, جھوٹ,
                                            &launch_event);
        اگر (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            خانہ.pending_event = launch_event;
            خانہ.has_pending   = سچ;
        }
    } پکڑو (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    اگر (ok) {
        uk_ctx->current_خانہ = (uk_ctx->current_خانہ + 1) % ggml_backend_et_uberkernel_context::خانہ_گنتی_2;
        خودکار & اگلا          = ggml_et_uberkernel_current_خانہ(uk_ctx);
        ggml_et_uberkernel_خانہ_wait(اگلا, runtime);
        اگلا.insts.clear();
        اگلا.params_blob.clear();
    } ورنہ {
        خانہ.insts.clear();
        خانہ.params_blob.clear();
    }
    uk_ctx->shire_نقاب = 0;
    واپس ok;
}

خالی ggml_et_uberkernel_abort_گراف(ggml_backend_et_uberkernel_context * uk_ctx) {
    اگر (!uk_ctx) {
        واپس;
    }

    uk_ctx->failed = جھوٹ;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

منطقی ggml_et_uberkernel_failed(ثابت ggml_backend_et_uberkernel_context * uk_ctx) {
    واپس uk_ctx && uk_ctx->failed;
}

ساکن منطقی ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ثابت std::string &              kernel_name,
                                      خالی *                           params,
                                      size_t                           params_سائز,
                                      uint64_t                         shire_نقاب,
                                      منطقی                             enable_print,
                                      منطقی                             sync_error_check) {
    اگر (!dev_ctx) {
        واپس جھوٹ;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ثابت uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    اگر (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        اگر (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            واپس جھوٹ;
        }
        واپس ggml_et_launch_kernel_اندرونی(dev_ctx, kernel_name, params, params_سائز, shire_نقاب, enable_print,
                                              sync_error_check);
    }

    خودکار &       خانہ          = ggml_et_uberkernel_current_خانہ(uk_ctx);
    ثابت size_t params_انحراف = ggml_et_align_اوپر(خانہ.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    اگر (params_انحراف > خانہ.params_blob.size()) {
        خانہ.params_blob.resize(params_انحراف);
    }

    ثابت std::byte * params_بائٹیں = reinterpret_cast<ثابت std::byte *>(params);
    خانہ.params_blob.insert(خانہ.params_blob.end(), params_بائٹیں, params_بائٹیں + params_سائز);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_انحراف),
        static_cast<uint32_t>(params_سائز),
    };
    خانہ.insts.push_back(inst);

    اگر (خانہ.insts.size() == 1) {
        uk_ctx->shire_نقاب = shire_نقاب;
    }

    واپس سچ;
}

منطقی ggml_et_uberkernel_انتہا_گراف(ggml_backend_et_device_context * dev_ctx) {
    اگر (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        واپس سچ;
    }

    واپس ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

منطقی ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ثابت std::string &              kernel_name,
                           خالی *                           params,
                           size_t                           params_سائز,
                           uint64_t                         shire_نقاب,
                           منطقی                             enable_print,
                           منطقی                             sync_error_check) {
    اگر (!dev_ctx) {
        واپس جھوٹ;
    }

    اگر (!dev_ctx->uberkernel_enabled) {
        واپس ggml_et_launch_kernel_اندرونی(dev_ctx, kernel_name, params, params_سائز, shire_نقاب, enable_print,
                                              sync_error_check);
    }

    واپس ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_سائز, shire_نقاب, enable_print,
                                     sync_error_check);
}

خالی ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        واپس;
    }

    خودکار kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    اگر (kernel_it != dev_ctx->loaded_kernels.end()) {
        کوشش {
            runtime->unload_رمز(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } پکڑو (ثابت std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

خالی ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    اگر (!dev_ctx) {
        واپس;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_نام;
    kernel_نام.reserve(dev_ctx->loaded_kernels.size());
    برائے (ثابت خودکار & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_نام.push_back(kernel_pair.پہلا);
    }

    برائے (ثابت خودکار & kernel_name : kernel_نام) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    برائے (ثابت خودکار & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    واپس loaded_kernels;
}

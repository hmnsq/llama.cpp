#ضمن "ggml-et-kernels.h"

#ضمن "ggml-et-kernels-embed.hpp"
#ضمن "ggml-et-uberkernel-kernel-map.h"
#ضمن "ggml-impl.h"

#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <fstream>

#تعریف_کرو ET_TRACE_DECODER_IMPL
#ضمن <et-trace/decoder.h>
#ضمن <et-trace/layout.h>

ساكن constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

ساكن size_t ggml_et_align_اتے(size_t value, size_t alignment) {
    أرجع (value + alignment - 1) & ~(alignment - 1);
}

ساكن size_t ggml_et_اگلا_سمرتھا(size_t current_سمرتھا, size_t required_سمرتھا) {
    إذا (current_سمرتھا == 0) {
        أرجع required_سمرتھا;
    }

    size_t اگلا_سمرتھا = current_سمرتھا;
    طالما (اگلا_سمرتھا < required_سمرتھا) {
        اگلا_سمرتھا *= 2;
    }

    أرجع اگلا_سمرتھا;
}

ساكن ggml_backend_et_uberkernel_تھاں & ggml_et_uberkernel_current_تھاں(ggml_backend_et_uberkernel_context * uk_ctx) {
    أرجع uk_ctx->slots[uk_ctx->current_تھاں];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
ساكن فراغ ggml_et_uberkernel_تھاں_wait(ggml_backend_et_uberkernel_تھاں &     تھاں,
                                         ثابت std::shared_ptr<rt::IRuntime> & runtime) {
    إذا (!تھاں.has_pending || !runtime) {
        أرجع;
    }
    runtime->waitForEvent(تھاں.pending_event);
    تھاں.has_pending = خطأ;
}

ساكن فراغ ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    إذا (!uk_ctx) {
        أرجع;
    }

    uk_ctx->shire_نقاب = 0;
    خودکار & تھاں        = ggml_et_uberkernel_current_تھاں(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_تھاں_wait(تھاں, ggml_et_runtime());
    تھاں.insts.clear();
    تھاں.params_blob.clear();
}

ساكن منطقي ggml_et_uberkernel_ensure_تھاں_سمرتھا(ggml_backend_et_uberkernel_تھاں & تھاں,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_ناپ,
                                                    size_t                            params_ناپ) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!dev_ctx || !runtime) {
        أرجع خطأ;
    }

    حاول {
        إذا (تھاں.device_insts == خالی_سوچک || insts_ناپ > تھاں.device_insts_سمرتھا) {
            ثابت size_t new_سمرتھا = ggml_et_اگلا_سمرتھا(تھاں.device_insts_سمرتھا, insts_ناپ);
            إذا (تھاں.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, تھاں.device_insts);
            }
            تھاں.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_سمرتھا);
            تھاں.device_insts_سمرتھا = تھاں.device_insts ? new_سمرتھا : 0;
        }

        إذا (تھاں.device_params == خالی_سوچک || params_ناپ > تھاں.device_params_سمرتھا) {
            ثابت size_t new_سمرتھا = ggml_et_اگلا_سمرتھا(تھاں.device_params_سمرتھا, params_ناپ);
            إذا (تھاں.device_params) {
                runtime->freeDevice(dev_ctx->rtid, تھاں.device_params);
            }
            تھاں.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_سمرتھا);
            تھاں.device_params_سمرتھا = تھاں.device_params ? new_سمرتھا : 0;
        }
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        أرجع خطأ;
    }

    أرجع تھاں.device_insts != خالی_سوچک && تھاں.device_params != خالی_سوچک;
}

// Get embedded kernel data by name
ساكن std::vector<std::byte> ggml_et_get_embedded_kernel(ثابت std::string & kernel_name) {
    خودکار it = ggml_et_embedded_kernels.find(kernel_name);
    إذا (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        أرجع {};
    }

    ثابت بنانشانی حرف * انکڑے = it->second.پہلا;
    uint64_t              size = it->second.second;

    std::vector<std::byte> بفر(size);
    std::حافظہ_نقل(بفر.انکڑے(), انکڑے, size);

    أرجع بفر;
}

// Read kernel from file (for development/override)
ساكن std::vector<std::byte> ggml_et_read_kernel_file(ثابت std::string & kernel_رستہ) {
    std::ifstream file(kernel_رستہ, std::ios::binary | std::ios::ate);
    إذا (!file) {
        أرجع {};
    }

    خودکار size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> بفر(size);
    file.اقرأ(reinterpret_cast<حرف *>(بفر.انکڑے()), size);

    أرجع بفر;
}

// Load kernel from file or embedded data
منطقي ggml_et_لدو_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        أرجع خطأ;
    }

    // Check if kernel already loaded
    إذا (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        أرجع سچ;
    }

    std::vector<std::byte> kernel_انکڑے;
    ثابت حرف *           kernels_رستہ = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    إذا (kernels_رستہ) {
        std::string kernel_file = std::string(kernels_رستہ) + "/" + kernel_name + ".elf";
        kernel_انکڑے             = ggml_et_read_kernel_file(kernel_file);

        إذا (!kernel_انکڑے.خالی_تھاواں()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } وإلا {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    إذا (kernel_انکڑے.خالی_تھاواں()) {
        kernel_انکڑے = ggml_et_get_embedded_kernel(kernel_name);
        إذا (kernel_انکڑے.خالی_تھاواں()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            أرجع خطأ;
        }
    }

    حاول {
        // Load kernel code using device's default stream
        خودکار لدو_نتیجہ = runtime->لدو_کوڈ(dev_ctx->default_دھارا, kernel_انکڑے.انکڑے(), kernel_انکڑے.size());
        runtime->waitForEvent(لدو_نتیجہ.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = لدو_نتیجہ.kernel_;
        أرجع سچ;

    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        أرجع خطأ;
    }
}

ساكن منطقي ggml_et_launch_kernel_اندرونی(ggml_backend_et_device_context * dev_ctx,
                                           ثابت std::string &              kernel_name,
                                           فراغ *                           params,
                                           size_t                           params_ناپ,
                                           uint64_t                         shire_نقاب,
                                           منطقي                             enable_print,
                                           منطقي                             sync_error_check,
                                           rt::EventId *                    باہرلا_event = خالی_سوچک) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        أرجع خطأ;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    خودکار kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    إذا (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        إذا (!ggml_et_لدو_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            أرجع خطأ;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        إذا (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            أرجع خطأ;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    حاول {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.سیٹ_Shire_نقاب(shire_نقاب);  // Default: all shires (0xFFFFFFFF)
        k_opts.سیٹ_Barrier(سچ);          // Wait for completion
        k_opts.سیٹ_Flush_L3(خطأ);         // No L3 flush needed
        إذا (enable_print) {
            k_opts.سیٹ_User_Tracing(reinterpret_cast<uint64_t>(dev_ctx->trace_بفر),
                                  static_cast<uint32_t>(ET_TRACE_بفر_ناپ),
                                  0,                      //حد
                                  shire_نقاب,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        إذا (sync_error_check) {
            runtime->wait_For_دھارا(dev_ctx->default_دھارا);
            خودکار errors = runtime->retrieve_دھارا_Errors(dev_ctx->default_دھارا);
            إذا (!errors.خالی_تھاواں()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                لكل (ثابت خودکار & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحيح) error.error_کوڈ);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_دھارا, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_ناپ, k_opts);
        إذا (باہرلا_event) {
            *باہرلا_event = launch_event;
        }

        إذا (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_بفر_ناپ);
            runtime->memcpy_Device_منزل_Host(dev_ctx->default_دھارا, dev_ctx->trace_بفر, host_trace_buf.انکڑے(),
                                        ET_TRACE_بفر_ناپ);
            runtime->wait_For_دھارا(dev_ctx->default_دھارا);
            ثابت خودکار * trace_سرخی = reinterpret_cast<ثابت trace_بفر_std_سرخی_t *>(host_trace_buf.انکڑے());
            ثابت trace_اندراج_سرخی_t * اندراج = خالی_سوچک;
            طالما ((اندراج = Trace_Decode(trace_سرخی, اندراج))) {
                إذا (اندراج->type != TRACE_TYPE_STRING) {
                    استمر;
                }
                ثابت خودکار * str_اندراج = reinterpret_cast<ثابت trace_string_t *>(اندراج);
                لکھو("[hart %d] %s", اندراج->hart_id, str_اندراج->string);
            }
        }

        إذا (sync_error_check) {
            // Already triggered. No need to retrigger
            إذا (!enable_print) {
                runtime->wait_For_دھارا(dev_ctx->default_دھارا);
            }
            خودکار errors = runtime->retrieve_دھارا_Errors(dev_ctx->default_دھارا);
            إذا (!errors.خالی_تھاواں()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                لكل (ثابت خودکار & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحيح) error.error_کوڈ);
                }
                abort();
            }
        }

        أرجع سچ;
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        أرجع خطأ;
    }
}

فراغ ggml_et_uberkernel_begin_گراف(ggml_backend_et_uberkernel_context * uk_ctx) {
    إذا (!uk_ctx) {
        أرجع;
    }

    uk_ctx->failed = خطأ;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

ساكن منطقي ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    إذا (!uk_ctx || !dev_ctx) {
        أرجع خطأ;
    }

    خودکار & تھاں = ggml_et_uberkernel_current_تھاں(uk_ctx);
    إذا (تھاں.insts.خالی_تھاواں()) {
        أرجع سچ;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = سچ;
        أرجع خطأ;
    }

    ثابت size_t   insts_ناپ  = تھاں.insts.size() * حجم(ggml_et_uberkernel_inst);
    ثابت size_t   params_ناپ = تھاں.params_blob.size();
    ثابت uint64_t shire_نقاب  = uk_ctx->shire_نقاب;
    منطقي           ok          = خطأ;

    حاول {
        إذا (!ggml_et_uberkernel_ensure_تھاں_سمرتھا(تھاں, dev_ctx, insts_ناپ, params_ناپ)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = سچ;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            تھاں.insts.clear();
            تھاں.params_blob.clear();
            uk_ctx->shire_نقاب = 0;
            أرجع خطأ;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpy_Host_منزل_Device(dev_ctx->default_دھارا, reinterpret_cast<ثابت std::byte *>(تھاں.insts.انکڑے()),
                                    تھاں.device_insts, insts_ناپ, سچ);
        runtime->memcpy_Host_منزل_Device(dev_ctx->default_دھارا, تھاں.params_blob.انکڑے(), تھاں.device_params, params_ناپ,
                                    سچ);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(تھاں.insts.size()),
            static_cast<uint32_t>(حجم(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(تھاں.device_insts),
            reinterpret_cast<uint64_t>(تھاں.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_اندرونی(dev_ctx, "uberkernel", &params, حجم(params), shire_نقاب, خطأ, خطأ,
                                            &launch_event);
        إذا (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            تھاں.pending_event = launch_event;
            تھاں.has_pending   = سچ;
        }
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    إذا (ok) {
        uk_ctx->current_تھاں = (uk_ctx->current_تھاں + 1) % ggml_backend_et_uberkernel_context::تھاں_گنتی_2;
        خودکار & اگلا          = ggml_et_uberkernel_current_تھاں(uk_ctx);
        ggml_et_uberkernel_تھاں_wait(اگلا, runtime);
        اگلا.insts.clear();
        اگلا.params_blob.clear();
    } وإلا {
        تھاں.insts.clear();
        تھاں.params_blob.clear();
    }
    uk_ctx->shire_نقاب = 0;
    أرجع ok;
}

فراغ ggml_et_uberkernel_abort_گراف(ggml_backend_et_uberkernel_context * uk_ctx) {
    إذا (!uk_ctx) {
        أرجع;
    }

    uk_ctx->failed = خطأ;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

منطقي ggml_et_uberkernel_failed(ثابت ggml_backend_et_uberkernel_context * uk_ctx) {
    أرجع uk_ctx && uk_ctx->failed;
}

ساكن منطقي ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      ثابت std::string &              kernel_name,
                                      فراغ *                           params,
                                      size_t                           params_ناپ,
                                      uint64_t                         shire_نقاب,
                                      منطقي                             enable_print,
                                      منطقي                             sync_error_check) {
    إذا (!dev_ctx) {
        أرجع خطأ;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    ثابت uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    إذا (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        إذا (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            أرجع خطأ;
        }
        أرجع ggml_et_launch_kernel_اندرونی(dev_ctx, kernel_name, params, params_ناپ, shire_نقاب, enable_print,
                                              sync_error_check);
    }

    خودکار &       تھاں          = ggml_et_uberkernel_current_تھاں(uk_ctx);
    ثابت size_t params_وتھ = ggml_et_align_اتے(تھاں.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    إذا (params_وتھ > تھاں.params_blob.size()) {
        تھاں.params_blob.resize(params_وتھ);
    }

    ثابت std::byte * params_بائٹاں = reinterpret_cast<ثابت std::byte *>(params);
    تھاں.params_blob.insert(تھاں.params_blob.end(), params_بائٹاں, params_بائٹاں + params_ناپ);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_وتھ),
        static_cast<uint32_t>(params_ناپ),
    };
    تھاں.insts.push_back(inst);

    إذا (تھاں.insts.size() == 1) {
        uk_ctx->shire_نقاب = shire_نقاب;
    }

    أرجع سچ;
}

منطقي ggml_et_uberkernel_سرا_گراف(ggml_backend_et_device_context * dev_ctx) {
    إذا (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        أرجع سچ;
    }

    أرجع ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

منطقي ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ثابت std::string &              kernel_name,
                           فراغ *                           params,
                           size_t                           params_ناپ,
                           uint64_t                         shire_نقاب,
                           منطقي                             enable_print,
                           منطقي                             sync_error_check) {
    إذا (!dev_ctx) {
        أرجع خطأ;
    }

    إذا (!dev_ctx->uberkernel_enabled) {
        أرجع ggml_et_launch_kernel_اندرونی(dev_ctx, kernel_name, params, params_ناپ, shire_نقاب, enable_print,
                                              sync_error_check);
    }

    أرجع ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_ناپ, shire_نقاب, enable_print,
                                     sync_error_check);
}

فراغ ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    خودکار kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    إذا (kernel_it != dev_ctx->loaded_kernels.end()) {
        حاول {
            runtime->unload_کوڈ(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } امسك (ثابت std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

فراغ ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    إذا (!dev_ctx) {
        أرجع;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_ناں;
    kernel_ناں.reserve(dev_ctx->loaded_kernels.size());
    لكل (ثابت خودکار & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_ناں.push_back(kernel_pair.پہلا);
    }

    لكل (ثابت خودکار & kernel_name : kernel_ناں) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    لكل (ثابت خودکار & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    أرجع loaded_kernels;
}

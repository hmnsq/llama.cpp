#ضمن "ggml-et-kernels.h"

#ضمن "ggml-et-kernels-embed.hpp"
#ضمن "ggml-et-uberkernel-kernel-map.h"
#ضمن "ggml-impl.h"

#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <fstream>

#وصف_ڪريو ET_TRACE_DECODER_IMPL
#ضمن <et-trace/decoder.h>
#ضمن <et-trace/layout.h>

ساكن constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

ساكن size_t ggml_et_align_مٿي(size_t value, size_t alignment) {
    أرجع (value + alignment - 1) & ~(alignment - 1);
}

ساكن size_t ggml_et_ايندڙ_صلاحيت(size_t current_صلاحيت, size_t required_صلاحيت) {
    إذا (current_صلاحيت == 0) {
        أرجع required_صلاحيت;
    }

    size_t ايندڙ_صلاحيت = current_صلاحيت;
    طالما (ايندڙ_صلاحيت < required_صلاحيت) {
        ايندڙ_صلاحيت *= 2;
    }

    أرجع ايندڙ_صلاحيت;
}

ساكن ggml_backend_et_uberkernel_جڳهه & ggml_et_uberkernel_current_جڳهه(ggml_backend_et_uberkernel_context * uk_ctx) {
    أرجع uk_ctx->slots[uk_ctx->current_جڳهه];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
ساكن فراغ ggml_et_uberkernel_جڳهه_wait(ggml_backend_et_uberkernel_جڳهه &     جڳهه,
                                         ثابت std::shared_ptr<rt::IRuntime> & runtime) {
    إذا (!جڳهه.has_pending || !runtime) {
        أرجع;
    }
    runtime->waitForEvent(جڳهه.pending_event);
    جڳهه.has_pending = خطأ;
}

ساكن فراغ ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    إذا (!uk_ctx) {
        أرجع;
    }

    uk_ctx->shire_نقاب = 0;
    خودڪار & جڳهه        = ggml_et_uberkernel_current_جڳهه(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_جڳهه_wait(جڳهه, ggml_et_runtime());
    جڳهه.insts.clear();
    جڳهه.params_blob.clear();
}

ساكن منطقي ggml_et_uberkernel_ensure_جڳهه_صلاحيت(ggml_backend_et_uberkernel_جڳهه & جڳهه,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_سائيز,
                                                    size_t                            params_سائيز) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!dev_ctx || !runtime) {
        أرجع خطأ;
    }

    حاول {
        إذا (جڳهه.device_insts == خالي_اشارو || insts_سائيز > جڳهه.device_insts_صلاحيت) {
            ثابت size_t new_صلاحيت = ggml_et_ايندڙ_صلاحيت(جڳهه.device_insts_صلاحيت, insts_سائيز);
            إذا (جڳهه.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, جڳهه.device_insts);
            }
            جڳهه.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_صلاحيت);
            جڳهه.device_insts_صلاحيت = جڳهه.device_insts ? new_صلاحيت : 0;
        }

        إذا (جڳهه.device_params == خالي_اشارو || params_سائيز > جڳهه.device_params_صلاحيت) {
            ثابت size_t new_صلاحيت = ggml_et_ايندڙ_صلاحيت(جڳهه.device_params_صلاحيت, params_سائيز);
            إذا (جڳهه.device_params) {
                runtime->freeDevice(dev_ctx->rtid, جڳهه.device_params);
            }
            جڳهه.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_صلاحيت);
            جڳهه.device_params_صلاحيت = جڳهه.device_params ? new_صلاحيت : 0;
        }
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        أرجع خطأ;
    }

    أرجع جڳهه.device_insts != خالي_اشارو && جڳهه.device_params != خالي_اشارو;
}

// Get embedded kernel data by name
ساكن std::vector<std::byte> ggml_et_get_embedded_kernel(ثابت std::string & kernel_name) {
    خودڪار it = ggml_et_embedded_kernels.find(kernel_name);
    إذا (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        أرجع {};
    }

    ثابت بنانشان حرف * معلومات = it->second.پهريون;
    uint64_t              size = it->second.second;

    std::vector<std::byte> بفر(size);
    std::يادگيري_نقل(بفر.معلومات(), معلومات, size);

    أرجع بفر;
}

// Read kernel from file (for development/override)
ساكن std::vector<std::byte> ggml_et_read_kernel_file(ثابت std::string & kernel_رستو) {
    std::ifstream file(kernel_رستو, std::ios::binary | std::ios::ate);
    إذا (!file) {
        أرجع {};
    }

    خودڪار size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> بفر(size);
    file.اقرأ(reinterpret_cast<حرف *>(بفر.معلومات()), size);

    أرجع بفر;
}

// Load kernel from file or embedded data
منطقي ggml_et_ڀريو_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
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

    std::vector<std::byte> kernel_معلومات;
    ثابت حرف *           kernels_رستو = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    إذا (kernels_رستو) {
        std::string kernel_file = std::string(kernels_رستو) + "/" + kernel_name + ".elf";
        kernel_معلومات             = ggml_et_read_kernel_file(kernel_file);

        إذا (!kernel_معلومات.خالي_جايون()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } وإلا {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    إذا (kernel_معلومات.خالي_جايون()) {
        kernel_معلومات = ggml_et_get_embedded_kernel(kernel_name);
        إذا (kernel_معلومات.خالي_جايون()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            أرجع خطأ;
        }
    }

    حاول {
        // Load kernel code using device's default stream
        خودڪار ڀريو_نتيجو = runtime->ڀريو_ڪوڊ(dev_ctx->default_وهڪرو, kernel_معلومات.معلومات(), kernel_معلومات.size());
        runtime->waitForEvent(ڀريو_نتيجو.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = ڀريو_نتيجو.kernel_;
        أرجع سچ;

    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        أرجع خطأ;
    }
}

ساكن منطقي ggml_et_launch_kernel_اندروني(ggml_backend_et_device_context * dev_ctx,
                                           ثابت std::string &              kernel_name,
                                           فراغ *                           params,
                                           size_t                           params_سائيز,
                                           uint64_t                         shire_نقاب,
                                           منطقي                             enable_print,
                                           منطقي                             sync_error_check,
                                           rt::EventId *                    ٻاهريون_event = خالي_اشارو) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        أرجع خطأ;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    خودڪار kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    إذا (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        إذا (!ggml_et_ڀريو_kernel(dev_ctx, kernel_name)) {
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
        k_opts.سيٽ_Shire_نقاب(shire_نقاب);  // Default: all shires (0xFFFFFFFF)
        k_opts.سيٽ_Barrier(سچ);          // Wait for completion
        k_opts.سيٽ_Flush_L3(خطأ);         // No L3 flush needed
        إذا (enable_print) {
            k_opts.سيٽ_User_Tracing(reinterpret_cast<uint64_t>(dev_ctx->trace_بفر),
                                  static_cast<uint32_t>(ET_TRACE_بفر_سائيز),
                                  0,                      //حد
                                  shire_نقاب,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        إذا (sync_error_check) {
            runtime->wait_For_وهڪرو(dev_ctx->default_وهڪرو);
            خودڪار errors = runtime->retrieve_وهڪرو_Errors(dev_ctx->default_وهڪرو);
            إذا (!errors.خالي_جايون()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                لكل (ثابت خودڪار & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحيح) error.error_ڪوڊ);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_وهڪرو, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_سائيز, k_opts);
        إذا (ٻاهريون_event) {
            *ٻاهريون_event = launch_event;
        }

        إذا (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_بفر_سائيز);
            runtime->memcpy_Device_منزل_Host(dev_ctx->default_وهڪرو, dev_ctx->trace_بفر, host_trace_buf.معلومات(),
                                        ET_TRACE_بفر_سائيز);
            runtime->wait_For_وهڪرو(dev_ctx->default_وهڪرو);
            ثابت خودڪار * trace_سرو = reinterpret_cast<ثابت trace_بفر_std_سرو_t *>(host_trace_buf.معلومات());
            ثابت trace_داخلا_سرو_t * داخلا = خالي_اشارو;
            طالما ((داخلا = Trace_Decode(trace_سرو, داخلا))) {
                إذا (داخلا->type != TRACE_TYPE_STRING) {
                    استمر;
                }
                ثابت خودڪار * str_داخلا = reinterpret_cast<ثابت trace_string_t *>(داخلا);
                لکو("[hart %d] %s", داخلا->hart_id, str_داخلا->string);
            }
        }

        إذا (sync_error_check) {
            // Already triggered. No need to retrigger
            إذا (!enable_print) {
                runtime->wait_For_وهڪرو(dev_ctx->default_وهڪرو);
            }
            خودڪار errors = runtime->retrieve_وهڪرو_Errors(dev_ctx->default_وهڪرو);
            إذا (!errors.خالي_جايون()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                لكل (ثابت خودڪار & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحيح) error.error_ڪوڊ);
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

    خودڪار & جڳهه = ggml_et_uberkernel_current_جڳهه(uk_ctx);
    إذا (جڳهه.insts.خالي_جايون()) {
        أرجع سچ;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = سچ;
        أرجع خطأ;
    }

    ثابت size_t   insts_سائيز  = جڳهه.insts.size() * حجم(ggml_et_uberkernel_inst);
    ثابت size_t   params_سائيز = جڳهه.params_blob.size();
    ثابت uint64_t shire_نقاب  = uk_ctx->shire_نقاب;
    منطقي           ok          = خطأ;

    حاول {
        إذا (!ggml_et_uberkernel_ensure_جڳهه_صلاحيت(جڳهه, dev_ctx, insts_سائيز, params_سائيز)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = سچ;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            جڳهه.insts.clear();
            جڳهه.params_blob.clear();
            uk_ctx->shire_نقاب = 0;
            أرجع خطأ;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpy_Host_منزل_Device(dev_ctx->default_وهڪرو, reinterpret_cast<ثابت std::byte *>(جڳهه.insts.معلومات()),
                                    جڳهه.device_insts, insts_سائيز, سچ);
        runtime->memcpy_Host_منزل_Device(dev_ctx->default_وهڪرو, جڳهه.params_blob.معلومات(), جڳهه.device_params, params_سائيز,
                                    سچ);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(جڳهه.insts.size()),
            static_cast<uint32_t>(حجم(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(جڳهه.device_insts),
            reinterpret_cast<uint64_t>(جڳهه.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_اندروني(dev_ctx, "uberkernel", &params, حجم(params), shire_نقاب, خطأ, خطأ,
                                            &launch_event);
        إذا (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            جڳهه.pending_event = launch_event;
            جڳهه.has_pending   = سچ;
        }
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    إذا (ok) {
        uk_ctx->current_جڳهه = (uk_ctx->current_جڳهه + 1) % ggml_backend_et_uberkernel_context::جڳهه_ڳڻپ_2;
        خودڪار & ايندڙ          = ggml_et_uberkernel_current_جڳهه(uk_ctx);
        ggml_et_uberkernel_جڳهه_wait(ايندڙ, runtime);
        ايندڙ.insts.clear();
        ايندڙ.params_blob.clear();
    } وإلا {
        جڳهه.insts.clear();
        جڳهه.params_blob.clear();
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
                                      size_t                           params_سائيز,
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
        أرجع ggml_et_launch_kernel_اندروني(dev_ctx, kernel_name, params, params_سائيز, shire_نقاب, enable_print,
                                              sync_error_check);
    }

    خودڪار &       جڳهه          = ggml_et_uberkernel_current_جڳهه(uk_ctx);
    ثابت size_t params_وٿي = ggml_et_align_مٿي(جڳهه.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    إذا (params_وٿي > جڳهه.params_blob.size()) {
        جڳهه.params_blob.resize(params_وٿي);
    }

    ثابت std::byte * params_بائيٽون = reinterpret_cast<ثابت std::byte *>(params);
    جڳهه.params_blob.insert(جڳهه.params_blob.end(), params_بائيٽون, params_بائيٽون + params_سائيز);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_وٿي),
        static_cast<uint32_t>(params_سائيز),
    };
    جڳهه.insts.push_back(inst);

    إذا (جڳهه.insts.size() == 1) {
        uk_ctx->shire_نقاب = shire_نقاب;
    }

    أرجع سچ;
}

منطقي ggml_et_uberkernel_پڇاڙي_گراف(ggml_backend_et_device_context * dev_ctx) {
    إذا (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        أرجع سچ;
    }

    أرجع ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

منطقي ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ثابت std::string &              kernel_name,
                           فراغ *                           params,
                           size_t                           params_سائيز,
                           uint64_t                         shire_نقاب,
                           منطقي                             enable_print,
                           منطقي                             sync_error_check) {
    إذا (!dev_ctx) {
        أرجع خطأ;
    }

    إذا (!dev_ctx->uberkernel_enabled) {
        أرجع ggml_et_launch_kernel_اندروني(dev_ctx, kernel_name, params, params_سائيز, shire_نقاب, enable_print,
                                              sync_error_check);
    }

    أرجع ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_سائيز, shire_نقاب, enable_print,
                                     sync_error_check);
}

فراغ ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    خودڪار kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    إذا (kernel_it != dev_ctx->loaded_kernels.end()) {
        حاول {
            runtime->unload_ڪوڊ(kernel_it->second);
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
    std::vector<std::string> kernel_نالا;
    kernel_نالا.reserve(dev_ctx->loaded_kernels.size());
    لكل (ثابت خودڪار & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_نالا.push_back(kernel_pair.پهريون);
    }

    لكل (ثابت خودڪار & kernel_name : kernel_نالا) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    لكل (ثابت خودڪار & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    أرجع loaded_kernels;
}

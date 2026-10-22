#ضمن "ggml-et-kernels.h"

#ضمن "ggml-et-kernels-embed.hpp"
#ضمن "ggml-et-uberkernel-kernel-map.h"
#ضمن "ggml-impl.h"

#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <fstream>

#تعریف_کن ET_TRACE_DECODER_IMPL
#ضمن <et-trace/decoder.h>
#ضمن <et-trace/layout.h>

ساكن constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

ساكن size_t ggml_et_align_به_بالا(size_t value, size_t alignment) {
    أرجع (value + alignment - 1) & ~(alignment - 1);
}

ساكن size_t ggml_et_بعدی_ظرفیت(size_t current_ظرفیت, size_t required_ظرفیت) {
    إذا (current_ظرفیت == 0) {
        أرجع required_ظرفیت;
    }

    size_t بعدی_ظرفیت = current_ظرفیت;
    طالما (بعدی_ظرفیت < required_ظرفیت) {
        بعدی_ظرفیت *= 2;
    }

    أرجع بعدی_ظرفیت;
}

ساكن ggml_backend_et_uberkernel_جایگاه & ggml_et_uberkernel_current_جایگاه(ggml_backend_et_uberkernel_context * uk_ctx) {
    أرجع uk_ctx->slots[uk_ctx->current_جایگاه];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
ساكن فراغ ggml_et_uberkernel_جایگاه_wait(ggml_backend_et_uberkernel_جایگاه &     جایگاه,
                                         ثابت std::shared_ptr<rt::IRuntime> & runtime) {
    إذا (!جایگاه.has_pending || !runtime) {
        أرجع;
    }
    runtime->waitForEvent(جایگاه.pending_event);
    جایگاه.has_pending = خطأ;
}

ساكن فراغ ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    إذا (!uk_ctx) {
        أرجع;
    }

    uk_ctx->shire_ماسک = 0;
    خودکار & جایگاه        = ggml_et_uberkernel_current_جایگاه(uk_ctx);
    // Drain any prior launch on this slot before clearing its host buffers.
    // begin_graph and abort_graph both come through here; in either case we
    // must not yank the source memory out from under an in-flight DMA.
    ggml_et_uberkernel_جایگاه_wait(جایگاه, ggml_et_runtime());
    جایگاه.insts.clear();
    جایگاه.params_blob.clear();
}

ساكن منطقي ggml_et_uberkernel_ensure_جایگاه_ظرفیت(ggml_backend_et_uberkernel_جایگاه & جایگاه,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_اندازه,
                                                    size_t                            params_اندازه) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!dev_ctx || !runtime) {
        أرجع خطأ;
    }

    حاول {
        إذا (جایگاه.device_insts == اشاره_تهی || insts_اندازه > جایگاه.device_insts_ظرفیت) {
            ثابت size_t new_ظرفیت = ggml_et_بعدی_ظرفیت(جایگاه.device_insts_ظرفیت, insts_اندازه);
            إذا (جایگاه.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, جایگاه.device_insts);
            }
            جایگاه.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_ظرفیت);
            جایگاه.device_insts_ظرفیت = جایگاه.device_insts ? new_ظرفیت : 0;
        }

        إذا (جایگاه.device_params == اشاره_تهی || params_اندازه > جایگاه.device_params_ظرفیت) {
            ثابت size_t new_ظرفیت = ggml_et_بعدی_ظرفیت(جایگاه.device_params_ظرفیت, params_اندازه);
            إذا (جایگاه.device_params) {
                runtime->freeDevice(dev_ctx->rtid, جایگاه.device_params);
            }
            جایگاه.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_ظرفیت);
            جایگاه.device_params_ظرفیت = جایگاه.device_params ? new_ظرفیت : 0;
        }
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        أرجع خطأ;
    }

    أرجع جایگاه.device_insts != اشاره_تهی && جایگاه.device_params != اشاره_تهی;
}

// Get embedded kernel data by name
ساكن std::vector<std::byte> ggml_et_get_embedded_kernel(ثابت std::string & kernel_name) {
    خودکار it = ggml_et_embedded_kernels.find(kernel_name);
    إذا (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        أرجع {};
    }

    ثابت بیعلامت حرف * داده = it->second.اولی;
    uint64_t              size = it->second.second;

    std::vector<std::byte> میانگیر(size);
    std::کپی_حافظه(میانگیر.داده(), داده, size);

    أرجع میانگیر;
}

// Read kernel from file (for development/override)
ساكن std::vector<std::byte> ggml_et_read_kernel_file(ثابت std::string & kernel_مسیر) {
    std::ifstream file(kernel_مسیر, std::ios::binary | std::ios::ate);
    إذا (!file) {
        أرجع {};
    }

    خودکار size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> میانگیر(size);
    file.اقرأ(reinterpret_cast<حرف *>(میانگیر.داده()), size);

    أرجع میانگیر;
}

// Load kernel from file or embedded data
منطقي ggml_et_بارگیری_kernel(ggml_backend_et_device_context * dev_ctx, ثابت std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        أرجع خطأ;
    }

    // Check if kernel already loaded
    إذا (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        أرجع درست;
    }

    std::vector<std::byte> kernel_داده;
    ثابت حرف *           kernels_مسیر = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    إذا (kernels_مسیر) {
        std::string kernel_file = std::string(kernels_مسیر) + "/" + kernel_name + ".elf";
        kernel_داده             = ggml_et_read_kernel_file(kernel_file);

        إذا (!kernel_داده.خالی()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } وإلا {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    إذا (kernel_داده.خالی()) {
        kernel_داده = ggml_et_get_embedded_kernel(kernel_name);
        إذا (kernel_داده.خالی()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            أرجع خطأ;
        }
    }

    حاول {
        // Load kernel code using device's default stream
        خودکار بارگیری_نتیجه = runtime->بارگیری_کد(dev_ctx->default_جریان, kernel_داده.داده(), kernel_داده.size());
        runtime->waitForEvent(بارگیری_نتیجه.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = بارگیری_نتیجه.kernel_;
        أرجع درست;

    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        أرجع خطأ;
    }
}

ساكن منطقي ggml_et_launch_kernel_داخلی(ggml_backend_et_device_context * dev_ctx,
                                           ثابت std::string &              kernel_name,
                                           فراغ *                           params,
                                           size_t                           params_اندازه,
                                           uint64_t                         shire_ماسک,
                                           منطقي                             enable_print,
                                           منطقي                             sync_error_check,
                                           rt::EventId *                    بیرون_event = اشاره_تهی) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        أرجع خطأ;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    خودکار kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    إذا (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        إذا (!ggml_et_بارگیری_kernel(dev_ctx, kernel_name)) {
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
        k_opts.مجموعه_Shire_ماسک(shire_ماسک);  // Default: all shires (0xFFFFFFFF)
        k_opts.مجموعه_Barrier(درست);          // Wait for completion
        k_opts.مجموعه_Flush_L3(خطأ);         // No L3 flush needed
        إذا (enable_print) {
            k_opts.مجموعه_User_Tracing(reinterpret_cast<uint64_t>(dev_ctx->trace_میانگیر),
                                  static_cast<uint32_t>(ET_TRACE_میانگیر_اندازه),
                                  0,                      //آستانه
                                  shire_ماسک,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        إذا (sync_error_check) {
            runtime->wait_For_جریان(dev_ctx->default_جریان);
            خودکار errors = runtime->retrieve_جریان_Errors(dev_ctx->default_جریان);
            إذا (!errors.خالی()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                لكل (ثابت خودکار & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحيح) error.error_کد);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_جریان, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_اندازه, k_opts);
        إذا (بیرون_event) {
            *بیرون_event = launch_event;
        }

        إذا (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_میانگیر_اندازه);
            runtime->memcpy_Device_مقصد_Host(dev_ctx->default_جریان, dev_ctx->trace_میانگیر, host_trace_buf.داده(),
                                        ET_TRACE_میانگیر_اندازه);
            runtime->wait_For_جریان(dev_ctx->default_جریان);
            ثابت خودکار * trace_سرآیند = reinterpret_cast<ثابت trace_میانگیر_std_سرآیند_t *>(host_trace_buf.داده());
            ثابت trace_مدخل_سرآیند_t * مدخل = اشاره_تهی;
            طالما ((مدخل = Trace_Decode(trace_سرآیند, مدخل))) {
                إذا (مدخل->type != TRACE_TYPE_STRING) {
                    استمر;
                }
                ثابت خودکار * str_مدخل = reinterpret_cast<ثابت trace_string_t *>(مدخل);
                چاپ_کن("[hart %d] %s", مدخل->hart_id, str_مدخل->string);
            }
        }

        إذا (sync_error_check) {
            // Already triggered. No need to retrigger
            إذا (!enable_print) {
                runtime->wait_For_جریان(dev_ctx->default_جریان);
            }
            خودکار errors = runtime->retrieve_جریان_Errors(dev_ctx->default_جریان);
            إذا (!errors.خالی()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                لكل (ثابت خودکار & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (صحيح) error.error_کد);
                }
                abort();
            }
        }

        أرجع درست;
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

    خودکار & جایگاه = ggml_et_uberkernel_current_جایگاه(uk_ctx);
    إذا (جایگاه.insts.خالی()) {
        أرجع درست;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = درست;
        أرجع خطأ;
    }

    ثابت size_t   insts_اندازه  = جایگاه.insts.size() * حجم(ggml_et_uberkernel_inst);
    ثابت size_t   params_اندازه = جایگاه.params_blob.size();
    ثابت uint64_t shire_ماسک  = uk_ctx->shire_ماسک;
    منطقي           ok          = خطأ;

    حاول {
        إذا (!ggml_et_uberkernel_ensure_جایگاه_ظرفیت(جایگاه, dev_ctx, insts_اندازه, params_اندازه)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = درست;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            جایگاه.insts.clear();
            جایگاه.params_blob.clear();
            uk_ctx->shire_ماسک = 0;
            أرجع خطأ;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpy_Host_مقصد_Device(dev_ctx->default_جریان, reinterpret_cast<ثابت std::byte *>(جایگاه.insts.داده()),
                                    جایگاه.device_insts, insts_اندازه, درست);
        runtime->memcpy_Host_مقصد_Device(dev_ctx->default_جریان, جایگاه.params_blob.داده(), جایگاه.device_params, params_اندازه,
                                    درست);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(جایگاه.insts.size()),
            static_cast<uint32_t>(حجم(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(جایگاه.device_insts),
            reinterpret_cast<uint64_t>(جایگاه.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_داخلی(dev_ctx, "uberkernel", &params, حجم(params), shire_ماسک, خطأ, خطأ,
                                            &launch_event);
        إذا (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            جایگاه.pending_event = launch_event;
            جایگاه.has_pending   = درست;
        }
    } امسك (ثابت std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    إذا (ok) {
        uk_ctx->current_جایگاه = (uk_ctx->current_جایگاه + 1) % ggml_backend_et_uberkernel_context::جایگاه_شمار_2;
        خودکار & بعدی          = ggml_et_uberkernel_current_جایگاه(uk_ctx);
        ggml_et_uberkernel_جایگاه_wait(بعدی, runtime);
        بعدی.insts.clear();
        بعدی.params_blob.clear();
    } وإلا {
        جایگاه.insts.clear();
        جایگاه.params_blob.clear();
    }
    uk_ctx->shire_ماسک = 0;
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
                                      size_t                           params_اندازه,
                                      uint64_t                         shire_ماسک,
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
        أرجع ggml_et_launch_kernel_داخلی(dev_ctx, kernel_name, params, params_اندازه, shire_ماسک, enable_print,
                                              sync_error_check);
    }

    خودکار &       جایگاه          = ggml_et_uberkernel_current_جایگاه(uk_ctx);
    ثابت size_t params_جابه_جایی = ggml_et_align_به_بالا(جایگاه.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    إذا (params_جابه_جایی > جایگاه.params_blob.size()) {
        جایگاه.params_blob.resize(params_جابه_جایی);
    }

    ثابت std::byte * params_بایتها = reinterpret_cast<ثابت std::byte *>(params);
    جایگاه.params_blob.insert(جایگاه.params_blob.end(), params_بایتها, params_بایتها + params_اندازه);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_جابه_جایی),
        static_cast<uint32_t>(params_اندازه),
    };
    جایگاه.insts.push_back(inst);

    إذا (جایگاه.insts.size() == 1) {
        uk_ctx->shire_ماسک = shire_ماسک;
    }

    أرجع درست;
}

منطقي ggml_et_uberkernel_انتها_گراف(ggml_backend_et_device_context * dev_ctx) {
    إذا (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        أرجع درست;
    }

    أرجع ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

منطقي ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           ثابت std::string &              kernel_name,
                           فراغ *                           params,
                           size_t                           params_اندازه,
                           uint64_t                         shire_ماسک,
                           منطقي                             enable_print,
                           منطقي                             sync_error_check) {
    إذا (!dev_ctx) {
        أرجع خطأ;
    }

    إذا (!dev_ctx->uberkernel_enabled) {
        أرجع ggml_et_launch_kernel_داخلی(dev_ctx, kernel_name, params, params_اندازه, shire_ماسک, enable_print,
                                              sync_error_check);
    }

    أرجع ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_اندازه, shire_ماسک, enable_print,
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
            runtime->unload_کد(kernel_it->second);
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
    std::vector<std::string> kernel_نامها;
    kernel_نامها.reserve(dev_ctx->loaded_kernels.size());
    لكل (ثابت خودکار & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_نامها.push_back(kernel_pair.اولی);
    }

    لكل (ثابت خودکار & kernel_name : kernel_نامها) {
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

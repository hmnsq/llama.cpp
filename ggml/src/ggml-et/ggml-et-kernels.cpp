#include "ggml-et-kernels.h"

#include "ggml-et-kernels-embed.hpp"
#include "ggml-et-uberkernel-kernel-map.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <fstream>

#define ET_TRACE_DECODER_IMPL
#include <et-trace/decoder.h>
#include <et-trace/layout.h>

static constexpr size_t GGML_ET_UBERKERNEL_PARAM_ALIGN = 64;

static size_t ggml_et_align_up(size_t value, size_t alignment) {
    ပွၵ်ႈ (value + alignment - 1) & ~(alignment - 1);
}

static size_t ggml_et_next_capacity(size_t current_capacity, size_t required_capacity) {
    ပေႃး (current_capacity == 0) {
        ပွၵ်ႈ required_capacity;
    }

    size_t next_capacity = current_capacity;
    ၶၢဝ်း (next_capacity < required_capacity) {
        next_capacity *= 2;
    }

    ပွၵ်ႈ next_capacity;
}

static ggml_backend_et_uberkernel_slot & ggml_et_uberkernel_current_slot(ggml_backend_et_uberkernel_context * uk_ctx) {
    ပွၵ်ႈ uk_ctx->slots[uk_ctx->current_slot];
}

// Wait for any in-flight launch that previously used this slot to finish,
// so the host vectors and device buffers are safe to mutate / free.
static void ggml_et_uberkernel_slot_wait(ggml_backend_et_uberkernel_slot &     slot,
                                         const std::shared_ptr<rt::IRuntime> & runtime) {
    ပေႃး (!slot.has_pending || !runtime) {
        ပွၵ်ႈ;
    }
    runtime->waitForEvent(slot.pending_event);
    slot.has_pending = false;
}

static void ggml_et_uberkernel_reset_segment(ggml_backend_et_uberkernel_context * uk_ctx) {
    ပေႃး (!uk_ctx) {
        ပွၵ်ႈ;
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

static bool ggml_et_uberkernel_ensure_slot_capacity(ggml_backend_et_uberkernel_slot & slot,
                                                    ggml_backend_et_device_context *  dev_ctx,
                                                    size_t                            insts_size,
                                                    size_t                            params_size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ပေႃး (!dev_ctx || !runtime) {
        ပွၵ်ႈ false;
    }

    try {
        ပေႃး (slot.device_insts == nullptr || insts_size > slot.device_insts_capacity) {
            const size_t new_capacity = ggml_et_next_capacity(slot.device_insts_capacity, insts_size);
            ပေႃး (slot.device_insts) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
            }
            slot.device_insts          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_insts_capacity = slot.device_insts ? new_capacity : 0;
        }

        ပေႃး (slot.device_params == nullptr || params_size > slot.device_params_capacity) {
            const size_t new_capacity = ggml_et_next_capacity(slot.device_params_capacity, params_size);
            ပေႃး (slot.device_params) {
                runtime->freeDevice(dev_ctx->rtid, slot.device_params);
            }
            slot.device_params          = runtime->mallocDevice(dev_ctx->rtid, new_capacity);
            slot.device_params_capacity = slot.device_params ? new_capacity : 0;
        }
    } gr_haa_na (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to resize uberkernel buffers: %s\n", e.what());
        ပွၵ်ႈ false;
    }

    ပွၵ်ႈ slot.device_insts != nullptr && slot.device_params != nullptr;
}

// Get embedded kernel data by name
static std::vector<std::byte> ggml_et_get_embedded_kernel(const std::string & kernel_name) {
    auto it = ggml_et_embedded_kernels.find(kernel_name);
    ပေႃး (it == ggml_et_embedded_kernels.end()) {
        GGML_LOG_ERROR("ET: Unknown embedded kernel: %s\n", kernel_name.c_str());
        ပွၵ်ႈ {};
    }

    const unsigned var_na * data = it->second.first;
    uint64_t              size = it->second.second;

    std::vector<std::byte> buffer(size);
    std::memcpy(buffer.data(), data, size);

    ပွၵ်ႈ buffer;
}

// Read kernel from file (for development/override)
static std::vector<std::byte> ggml_et_read_kernel_file(const std::string & kernel_path) {
    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    ပေႃး (!file) {
        ပွၵ်ႈ {};
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<std::byte> buffer(size);
    file.read(reinterpret_cast<var_na *>(buffer.data()), size);

    ပွၵ်ႈ buffer;
}

// Load kernel from file or embedded data
bool ggml_et_load_kernel(ggml_backend_et_device_context * dev_ctx, const std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ပေႃး (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel loading\n");
        ပွၵ်ႈ false;
    }

    // Check if kernel already loaded
    ပေႃး (dev_ctx->loaded_kernels.find(kernel_name) != dev_ctx->loaded_kernels.end()) {
        GGML_LOG_DEBUG("ET: Kernel %s already loaded on device %d\n", kernel_name.c_str(), dev_ctx->devidx);
        ပွၵ်ႈ true;
    }

    std::vector<std::byte> kernel_data;
    const var_na *           kernels_path = getenv("GGML_ET_KERNELS_PATH");

    // If GGML_ET_KERNELS_PATH is set, try to load from file first
    ပေႃး (kernels_path) {
        std::string kernel_file = std::string(kernels_path) + "/" + kernel_name + ".elf";
        kernel_data             = ggml_et_read_kernel_file(kernel_file);

        ပေႃး (!kernel_data.empty()) {
            GGML_LOG_INFO("ET: Loading kernel %s from file: %s\n", kernel_name.c_str(), kernel_file.c_str());
        } ပေႃးဢမ်ႇ {
            GGML_LOG_INFO("ET: Kernel file not found: %s, falling back to embedded\n", kernel_file.c_str());
        }
    }

    // If no file data, use embedded kernel
    ပေႃး (kernel_data.empty()) {
        kernel_data = ggml_et_get_embedded_kernel(kernel_name);
        ပေႃး (kernel_data.empty()) {
            GGML_LOG_ERROR("ET: Failed to get kernel data for %s\n", kernel_name.c_str());
            ပွၵ်ႈ false;
        }
    }

    try {
        // Load kernel code using device's default stream
        auto load_result = runtime->loadCode(dev_ctx->default_stream, kernel_data.data(), kernel_data.size());
        runtime->waitForEvent(load_result.event_);

        // Store kernel handle
        dev_ctx->loaded_kernels[kernel_name] = load_result.kernel_;
        ပွၵ်ႈ true;

    } gr_haa_na (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to load kernel %s: %s\n", kernel_name.c_str(), e.what());
        ပွၵ်ႈ false;
    }
}

static bool ggml_et_launch_kernel_internal(ggml_backend_et_device_context * dev_ctx,
                                           const std::string &              kernel_name,
                                           void *                           params,
                                           size_t                           params_size,
                                           uint64_t                         shire_mask,
                                           bool                             enable_print,
                                           bool                             sync_error_check,
                                           rt::EventId *                    out_event = nullptr) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ပေႃး (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for kernel launch\n");
        ပွၵ်ႈ false;
    }

    // Lazy loading: check if kernel is loaded, load if needed
    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ပေႃး (kernel_it == dev_ctx->loaded_kernels.end()) {
        // Kernel not loaded - load it
        ပေႃး (!ggml_et_load_kernel(dev_ctx, kernel_name)) {
            GGML_LOG_ERROR("ET: Failed to lazy-load kernel %s\n", kernel_name.c_str());
            ပွၵ်ႈ false;
        }

        // Update iterator after successful load
        kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
        ပေႃး (kernel_it == dev_ctx->loaded_kernels.end()) {
            GGML_LOG_ERROR("ET: Kernel %s not found after loading\n", kernel_name.c_str());
            ပွၵ်ႈ false;
        }
    }

    rt::KernelId kernel_id = kernel_it->second;

    try {
        // Setup kernel launch options
        rt::KernelLaunchOptions k_opts;
        k_opts.setShireMask(shire_mask);  // Default: all shires (0xFFFFFFFF)
        k_opts.setBarrier(true);          // Wait for completion
        k_opts.setFlushL3(false);         // No L3 flush needed
        ပေႃး (enable_print) {
            k_opts.setUserTracing(reinterpret_cast<uint64_t>(dev_ctx->trace_buffer),
                                  static_cast<uint32_t>(ET_TRACE_BUFFER_SIZE),
                                  0,                      // threshold
                                  shire_mask,             // shire mask
                                  0xFFFFFFFFFFFFFFFFULL,  // threadMask - all threads
                                  0xFFFFFFFFU,            // eventMask - all events
                                  0xFFFFFFFFU             // filterMask - all levels
            );
        }

        ပေႃး (sync_error_check) {
            runtime->waitForStream(dev_ctx->default_stream);
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            ပေႃး (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected before kernel \"%s\" launch\n", kernel_name.c_str());
                for (const auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (puur_naa_nka) error.errorCode_);
                }
                abort();
            }
        }

        rt::EventId launch_event = runtime->kernelLaunch(dev_ctx->default_stream, kernel_id,
                                                         reinterpret_cast<std::byte *>(params), params_size, k_opts);
        ပေႃး (out_event) {
            *out_event = launch_event;
        }

        ပေႃး (enable_print) {
            std::vector<std::byte> host_trace_buf(ET_TRACE_BUFFER_SIZE);
            runtime->memcpyDeviceToHost(dev_ctx->default_stream, dev_ctx->trace_buffer, host_trace_buf.data(),
                                        ET_TRACE_BUFFER_SIZE);
            runtime->waitForStream(dev_ctx->default_stream);
            const auto * trace_header = reinterpret_cast<const trace_buffer_std_header_t *>(host_trace_buf.data());
            const trace_entry_header_t * entry = nullptr;
            ၶၢဝ်း ((entry = Trace_Decode(trace_header, entry))) {
                ပေႃး (entry->type != TRACE_TYPE_STRING) {
                    continue;
                }
                const auto * str_entry = reinterpret_cast<const trace_string_t *>(entry);
                printf("[hart %d] %s", entry->hart_id, str_entry->string);
            }
        }

        ပေႃး (sync_error_check) {
            // Already triggered. No need to retrigger
            ပေႃး (!enable_print) {
                runtime->waitForStream(dev_ctx->default_stream);
            }
            auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
            ပေႃး (!errors.empty()) {
                GGML_LOG_ERROR("ET: Errors detected during kernel \"%s\" execution\n", kernel_name.c_str());
                for (const auto & error : errors) {
                    GGML_LOG_ERROR("ET: Error code: %d\n", (puur_naa_nka) error.errorCode_);
                }
                abort();
            }
        }

        ပွၵ်ႈ true;
    } gr_haa_na (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to launch kernel %s: %s\n", kernel_name.c_str(), e.what());
        ပွၵ်ႈ false;
    }
}

void ggml_et_uberkernel_begin_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    ပေႃး (!uk_ctx) {
        ပွၵ်ႈ;
    }

    uk_ctx->failed = false;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

static bool ggml_et_launch_uberkernel_segment(ggml_backend_et_device_context *     dev_ctx,
                                              ggml_backend_et_uberkernel_context * uk_ctx) {
    ပေႃး (!uk_ctx || !dev_ctx) {
        ပွၵ်ႈ false;
    }

    auto & slot = ggml_et_uberkernel_current_slot(uk_ctx);
    ပေႃး (slot.insts.empty()) {
        ပွၵ်ႈ true;
    }

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ပေႃး (!runtime) {
        GGML_LOG_ERROR("ET: Runtime not available for uberkernel commit\n");
        uk_ctx->failed = true;
        ပွၵ်ႈ false;
    }

    const size_t   insts_size  = slot.insts.size() * parimaa_na(ggml_et_uberkernel_inst);
    const size_t   params_size = slot.params_blob.size();
    const uint64_t shire_mask  = uk_ctx->shire_mask;
    bool           ok          = false;

    try {
        ပေႃး (!ggml_et_uberkernel_ensure_slot_capacity(slot, dev_ctx, insts_size, params_size)) {
            GGML_LOG_ERROR("ET: Failed to allocate uberkernel device buffers\n");
            uk_ctx->failed = true;
            // Drop this segment but keep the slot drained so we don't leak
            // host vectors into the next graph.
            slot.insts.clear();
            slot.params_blob.clear();
            uk_ctx->shire_mask = 0;
            ပွၵ်ႈ false;
        }

        // Fire-and-forget H2D + launch on default_stream. In-stream FIFO
        // ordering guarantees the kernel sees fully-uploaded buffers; the
        // host source bytes (slot.insts / slot.params_blob) stay alive
        // because we won't touch this slot again until pending_event fires.
        runtime->memcpyHostToDevice(dev_ctx->default_stream, reinterpret_cast<const std::byte *>(slot.insts.data()),
                                    slot.device_insts, insts_size, true);
        runtime->memcpyHostToDevice(dev_ctx->default_stream, slot.params_blob.data(), slot.device_params, params_size,
                                    true);

        ggml_et_uberkernel_params params = {
            static_cast<uint32_t>(slot.insts.size()),
            static_cast<uint32_t>(parimaa_na(ggml_et_uberkernel_inst)),
            reinterpret_cast<uint64_t>(slot.device_insts),
            reinterpret_cast<uint64_t>(slot.device_params),
        };

        rt::EventId launch_event{};
        ok = ggml_et_launch_kernel_internal(dev_ctx, "uberkernel", &params, parimaa_na(params), shire_mask, false, false,
                                            &launch_event);
        ပေႃး (ok) {
            // The kernelLaunch above is the last thing on default_stream
            // that touches this slot's device buffers. Recording its event
            // lets the next reuse of this slot wait on that one event
            // instead of the whole stream.
            slot.pending_event = launch_event;
            slot.has_pending   = true;
        }
    } gr_haa_na (const std::exception & e) {
        GGML_LOG_ERROR("ET: Failed to commit uberkernel segment: %s\n", e.what());
    }
    uk_ctx->failed = !ok;

    ပေႃး (ok) {
        uk_ctx->current_slot = (uk_ctx->current_slot + 1) % ggml_backend_et_uberkernel_context::SLOT_COUNT;
        auto & next          = ggml_et_uberkernel_current_slot(uk_ctx);
        ggml_et_uberkernel_slot_wait(next, runtime);
        next.insts.clear();
        next.params_blob.clear();
    } ပေႃးဢမ်ႇ {
        slot.insts.clear();
        slot.params_blob.clear();
    }
    uk_ctx->shire_mask = 0;
    ပွၵ်ႈ ok;
}

void ggml_et_uberkernel_abort_graph(ggml_backend_et_uberkernel_context * uk_ctx) {
    ပေႃး (!uk_ctx) {
        ပွၵ်ႈ;
    }

    uk_ctx->failed = false;
    ggml_et_uberkernel_reset_segment(uk_ctx);
}

bool ggml_et_uberkernel_failed(const ggml_backend_et_uberkernel_context * uk_ctx) {
    ပွၵ်ႈ uk_ctx && uk_ctx->failed;
}

static bool ggml_et_launch_uberkernel(ggml_backend_et_device_context * dev_ctx,
                                      const std::string &              kernel_name,
                                      void *                           params,
                                      size_t                           params_size,
                                      uint64_t                         shire_mask,
                                      bool                             enable_print,
                                      bool                             sync_error_check) {
    ပေႃး (!dev_ctx) {
        ပွၵ်ႈ false;
    }

    ggml_backend_et_uberkernel_context * uk_ctx        = &dev_ctx->uberkernel;
    const uint16_t                       uberkernel_id = ggml_et_uberkernel_kernel_id_from_name(kernel_name.c_str());
    ပေႃး (uberkernel_id == GGML_ET_UBERKERNEL_KERNEL_INVALID) {
        ပေႃး (!ggml_et_launch_uberkernel_segment(dev_ctx, uk_ctx)) {
            ပွၵ်ႈ false;
        }
        ပွၵ်ႈ ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    auto &       slot          = ggml_et_uberkernel_current_slot(uk_ctx);
    const size_t params_offset = ggml_et_align_up(slot.params_blob.size(), GGML_ET_UBERKERNEL_PARAM_ALIGN);
    ပေႃး (params_offset > slot.params_blob.size()) {
        slot.params_blob.resize(params_offset);
    }

    const std::byte * params_bytes = reinterpret_cast<const std::byte *>(params);
    slot.params_blob.insert(slot.params_blob.end(), params_bytes, params_bytes + params_size);

    ggml_et_uberkernel_inst inst = {
        uberkernel_id,
        0,
        static_cast<uint32_t>(params_offset),
        static_cast<uint32_t>(params_size),
    };
    slot.insts.push_back(inst);

    ပေႃး (slot.insts.size() == 1) {
        uk_ctx->shire_mask = shire_mask;
    }

    ပွၵ်ႈ true;
}

bool ggml_et_uberkernel_end_graph(ggml_backend_et_device_context * dev_ctx) {
    ပေႃး (!dev_ctx || !dev_ctx->uberkernel_enabled) {
        ပွၵ်ႈ true;
    }

    ပွၵ်ႈ ggml_et_launch_uberkernel_segment(dev_ctx, &dev_ctx->uberkernel);
}

bool ggml_et_launch_kernel(ggml_backend_et_device_context * dev_ctx,
                           const std::string &              kernel_name,
                           void *                           params,
                           size_t                           params_size,
                           uint64_t                         shire_mask,
                           bool                             enable_print,
                           bool                             sync_error_check) {
    ပေႃး (!dev_ctx) {
        ပွၵ်ႈ false;
    }

    ပေႃး (!dev_ctx->uberkernel_enabled) {
        ပွၵ်ႈ ggml_et_launch_kernel_internal(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                              sync_error_check);
    }

    ပွၵ်ႈ ggml_et_launch_uberkernel(dev_ctx, kernel_name, params, params_size, shire_mask, enable_print,
                                     sync_error_check);
}

void ggml_et_unload_kernel(ggml_backend_et_device_context * dev_ctx, const std::string & kernel_name) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ပေႃး (!runtime) {
        ပွၵ်ႈ;
    }

    auto kernel_it = dev_ctx->loaded_kernels.find(kernel_name);
    ပေႃး (kernel_it != dev_ctx->loaded_kernels.end()) {
        try {
            runtime->unloadCode(kernel_it->second);
            dev_ctx->loaded_kernels.erase(kernel_it);
        } gr_haa_na (const std::exception & e) {
            GGML_LOG_ERROR("ET: Failed to unload kernel %s: %s\n", kernel_name.c_str(), e.what());
        }
    }
}

void ggml_et_unload_all_kernels(ggml_backend_et_device_context * dev_ctx) {
    ပေႃး (!dev_ctx) {
        ပွၵ်ႈ;
    }

    // Make a copy of kernel names since ggml_et_unload_kernel modifies the map
    std::vector<std::string> kernel_names;
    kernel_names.reserve(dev_ctx->loaded_kernels.size());
    for (const auto & kernel_pair : dev_ctx->loaded_kernels) {
        kernel_names.push_back(kernel_pair.first);
    }

    for (const auto & kernel_name : kernel_names) {
        ggml_et_unload_kernel(dev_ctx, kernel_name);
    }
}

std::vector<std::pair<std::string, rt::KernelId>> ggml_et_get_loaded_kernels(ggml_backend_et_device_context * dev_ctx) {
    std::vector<std::pair<std::string, rt::KernelId>> loaded_kernels;
    loaded_kernels.reserve(dev_ctx->loaded_kernels.size());
    for (const auto & kernel_pair : dev_ctx->loaded_kernels) {
        loaded_kernels.push_back(kernel_pair);
    }
    ပွၵ်ႈ loaded_kernels;
}

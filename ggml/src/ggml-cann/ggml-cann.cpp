/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#অন্তর্ভাব "ggml-cann.h"

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-cann/aclnn_ops.h"
#অন্তর্ভাব "ggml-cann/common.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <acl/acl.h>
#অন্তর্ভাব <aclnnop/aclnn_trans_matmul_weight.h>
#অন্তর্ভাব <stdarg.h>

#অন্তর্ভাব <chrono>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <optional>
#অন্তর্ভাব <queue>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <vector>

#define GGML_COMMON_DECL_C

#অন্তর্ভাব "ggml-common.h"

#define GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] শূন্য ggml_cann_error(ধ্রুব বর্ণ * stmt, ধ্রুব বর্ণ * func, ধ্রুব বর্ণ * file, পূর্ণাঙ্ক সারি, ধ্রুব বর্ণ * বার্তা) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", বার্তা);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, সারি);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

// Thread-local variable to record the current device of this thread.
thread_local পূর্ণাঙ্ক g_current_cann_device = -1;

/**
 * @brief Set the CANN device to be used.
 *
 * @param device The target device ID to set.
 */
শূন্য ggml_cann_set_device(ধ্রুব int32_t device) {
    // int current_device = -1;
    // Note: In some CANN versions, if no device has been set yet,
    //       aclrtGetDevice(&current_device) may return 0 by default.
    // aclrtGetDevice(&current_device);

    // If the current device is already the target one, no need to switch.
    যদি (device == g_current_cann_device) {
        প্রত্যাগম;
    }

    // Switch to the new device.
    ACL_CHECK(aclrtSetDevice(device));

    // Update the global device record.
    g_current_cann_device = device;
}

/**
 * @brief Get the value of the specified environment variable (name) as lowercase.
 *        if not empty, return a std::string object
 */
std::optional<std::string> get_env_as_lowercase(ধ্রুব std::string & name) {
    ধ্রুব বর্ণ * val = std::getenv(name.c_str());
    যদি (!val) {
        প্রত্যাগম std::nullopt;
    }
    std::string res = std::string(val);
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    প্রত্যাগম res;
}

/**
 * @brief Verify whether the environment variable is a valid value.
 */
তার্কিক parse_bool(ধ্রুব std::string & value) {
    স্থির ধ্রুব std::unordered_set<std::string> valid_মানসমূহ = { "on", "1", "yes", "y", "enable", "true" };
    প্রত্যাগম valid_মানসমূহ.find(value) != valid_মানসমূহ.end();
}

/**
 * @brief Parse a string as an integer, returning 0 if invalid.
 *
 * This function attempts to convert the input string `value` to an `int`.
 * If the string is not a valid integer or is out of the `int` range,
 * it returns 0.
 *
 * @param value The string to parse.
 * @return The parsed integer, or 0 if conversion fails.
 */
পূর্ণাঙ্ক parse_integer(ধ্রুব std::string & value) {
    যত্ন {
        প্রত্যাগম std::stoi(value);
    } গৃহাণ (...) {
        প্রত্যাগম 0;
    }
}

/**
 * @brief Initialize the CANN device information.
 *
 * This function initializes the CANN device information by obtaining the
 * device count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
স্থির ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrt_Get_Device_গুনতি((uint32_t *) &info.device_গুনতি);

    যদি (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n", __func__, acl_Get_Recent_Err_বার্তা());
        প্রত্যাগম info;
    }

    GGML_ASSERT(info.device_গুনতি <= GGML_CANN_MAX_DEVICES);

    প্রতি (পূর্ণাঙ্ক id = 0; id < info.device_গুনতি; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType       = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr              = ACL_HBM_MEM_HUGE;
        prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = id;
        prop.reserve              = 0;
        err                       = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                                                                     &info.devices[id].vmm_granularity);
        info.devices[id].vmm      = err == ACL_SUCCESS;

        size_t মুক্ত, total;
        ggml_backend_cann_get_device_memory(id, &মুক্ত, &total);
        info.devices[id].total_vram = মুক্ত;
    }

    // TODO: add more device info later.
    প্রত্যাগম info;
}

/**
 * @brief Retrieve the CANN device information.
 *
 * This function returns a reference to a structure containing the CANN device
 * information. The device information is initialized once and reused on
 * subsequent calls.
 *
 * @return A reference to the structure containing the device information.
 */
ধ্রুব ggml_cann_device_info & ggml_cann_info() {
    স্থির ggml_cann_device_info info = ggml_cann_init();
    প্রত্যাগম info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(priority segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
সংরচনা ggml_cann_pool_buf_prio : সার্বজনিক ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    স্থির ধ্রুব size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    স্থির ধ্রুব size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    স্থির ধ্রুব size_t alignment = 128;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    পূর্ণাঙ্ক device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    তার্কিক disable_clean = অসত্য;

    /**
     * @brief Structure representing a CANN buffer.
     */
    সংরচনা ggml_cann_buffer {
        শূন্য *                                ptr  = nullptr;  ///< Pointer to the buffer.
        size_t                                size = 0;        ///< Size of the buffer.
        std::chrono::steady_clock::time_point last_used;       ///< Last used time.

        তার্কিক operator>(ধ্রুব ggml_cann_buffer & other) ধ্রুব { প্রত্যাগম size > other.size; }
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    std::unordered_map<শূন্য *, size_t>                                                   buffer_pool;
    std::priority_queue<ggml_cann_buffer, std::vector<ggml_cann_buffer>, std::greater<>> free_buffers;

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_buf_prio(পূর্ণাঙ্ক device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf_prio() {
        ggml_cann_set_device(device);
        প্রতি (auto & [b_ptr, b_size] : buffer_pool) {
            aclrtFree(b_ptr);
            pool_size -= b_size;
        }
        buffer_pool.clear();
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    শূন্য * alloc(size_t size, size_t * actual_size) override {
        size = GGML_PAD(size, alignment);
        যদি (size == 0) {
            size = alignment;
        }

        শূন্য * ptr = nullptr;
        auto   now = std::chrono::steady_clock::now();

        std::vector<ggml_cann_buffer> free_buffers_rest;
        free_buffers_rest.reserve(free_buffers.size());
        যাবত্ (!free_buffers.empty()) {
            auto b = free_buffers.top();
            free_buffers.বের_করো();

            যদি (b.size >= size) {
                // reuse the buffer if the size is enough
                ধ্রুব size_t margin = b.size - size;
                যদি (margin <= max_reuse_margin) {
                    *actual_size = b.size;
                    ptr          = b.ptr;
#ifdef DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#endif
                    ভঙ্গ;
                }
            }

            তার্কিক should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            যদি (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
                buffer_pool.erase(b.ptr);
#ifdef DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#endif
                অনুবর্ত;
            }
            free_buffers_rest.push_back(b);
        }
        প্রতি (ggml_cann_buffer & b : free_buffers_rest) {
            free_buffers.ঠেলো(std::move(b));
        }

#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d] free pool_size = %5u MB\n\n", device,
                      (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576));
#endif
        যদি (ptr != nullptr) {
            প্রত্যাগম ptr;
        }

        // allocate a new buffer if no buffer can be reused
        ggml_cann_set_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        *actual_size = size;
        pool_size += size;
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: allocate %p, "
            "pool_size = %5u MB, "
            "size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
            (uint32_t) (GGML_PAD(size, 1048576) / 1048576));
#endif
        buffer_pool.emplace(ptr, size);
        প্রত্যাগম ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    শূন্য মুক্ত(শূন্য * ptr, size_t size) override {
        GGML_UNUSED(size);
        auto it = buffer_pool.find(ptr);
        যদি (it == buffer_pool.end()) {
            GGML_ABORT("cann pool[%d]: buffer %p not found in pool\n", device, ptr);
        }

        auto now = std::chrono::steady_clock::now();
        free_buffers.emplace(ggml_cann_buffer{ ptr, it->second, now });
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: return   %p, "
            "pool_size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576));
#endif
    }
};

/**
 * @brief A pool of CANN buffers(segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
সংরচনা ggml_cann_pool_buf : সার্বজনিক ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    স্থির ধ্রুব size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    স্থির ধ্রুব size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    স্থির ধ্রুব size_t alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    স্থির ধ্রুব পূর্ণাঙ্ক MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    পূর্ণাঙ্ক device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    তার্কিক disable_clean = অসত্য;

    /**
     * @brief Structure representing a CANN buffer.
     */
    সংরচনা ggml_cann_buffer {
        শূন্য *                                ptr  = nullptr;  ///< Pointer to the buffer memory.
        size_t                                size = 0;        ///< Size of the buffer.
        তার্কিক                                  used = অসত্য;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point last_used;       ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_buffer buffer_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_buf(পূর্ণাঙ্ক device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf() {
        ggml_cann_set_device(device);
        প্রতি (পূর্ণাঙ্ক i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer & b = buffer_pool[i];
            যদি (b.ptr != nullptr) {
                aclrtFree(b.ptr);
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    শূন্য * alloc(size_t size, size_t * actual_size) override {
        size = GGML_PAD(size, alignment);
        যদি (size == 0) {
            size = alignment;
        }

        শূন্য * ptr = nullptr;
        auto   now = std::chrono::steady_clock::now();

        পূর্ণাঙ্ক i = 0;
        প্রতি (; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer & b = buffer_pool[i];
            যদি (b.ptr == nullptr) {
                ভঙ্গ;
            }
            যদি (b.used) {
                অনুবর্ত;
            }
            যদি (b.size >= size) {
                // reuse the buffer if the size is enough
                ধ্রুব size_t margin = b.size - size;
                যদি (margin <= max_reuse_margin) {
                    *actual_size = b.size;
                    b.used       = সত্য;
                    ptr          = b.ptr;
#ifdef DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#endif
                    ভঙ্গ;
                }
            }

            তার্কিক should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            যদি (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
#ifdef DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#endif
                b.ptr = nullptr;
            }
        }
        যদি (ptr != nullptr) {
            প্রত্যাগম ptr;
        }

        যদি (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_cann_buffer & b = buffer_pool[i];
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtMalloc(&b.ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
            pool_size += size;
            *actual_size = size;
            b.size       = size;
            b.used       = সত্য;
            যদি (i >= MAX_BUFFERS - 8) {
                GGML_LOG_WARN("cann pool[%d]: slots almost full\n", device);
            }
#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#endif
            প্রত্যাগম b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    শূন্য মুক্ত(শূন্য * ptr, size_t size) override {
        GGML_UNUSED(size);
        প্রতি (পূর্ণাঙ্ক i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer & b = buffer_pool[i];
            যদি (b.ptr != ptr) {
                অনুবর্ত;
            }
            b.used      = অসত্য;
            b.last_used = std::chrono::steady_clock::now();
#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576));
#endif
            প্রত্যাগম;
        }
        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }
};

/**
 * @brief A pool of CANN buffers with virtual memory.
 *
 * This class manages a pool of CANN buffers with virtual memory for a specific
 * device.
 */
সংরচনা ggml_cann_pool_vmm : সার্বজনিক ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    size_t max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    পূর্ণাঙ্ক device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    শূন্য * pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_used = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Allocation granularity for the virtual memory pool.
     */
    size_t granularity;

    /**
     * @brief Handles for the physical memory allocated.
     */
    std::vector<aclrtDrvMemHandle> handles;

    /**
     * @brief Offsets for the mapped memory regions.
     */
    std::vector<শূন্য *> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_vmm(পূর্ণাঙ্ক device) : device(device) {
        auto dev    = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size    = dev.total_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        যদি (pool_addr != 0) {
            প্রতি (auto & offset : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(offset));
            }
            প্রতি (auto & handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrtReleaseMemAddress(pool_addr));
        }
    }

    /**
     * @brief Allocate a buffer of the given size in the virtual memory pool.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    শূন্য * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        ধ্রুব size_t alignment = 128;
        size                   = GGML_PAD(size, alignment);
        যদি (size == 0) {
            size = alignment;
        }

        size_t avail = pool_size - pool_used;

        যদি (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size        = GGML_PAD(reserve_size, granularity);

            GGML_ASSERT(pool_size + reserve_size <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType       = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr              = ACL_HBM_MEM_HUGE;
            prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id          = device;
            prop.reserve              = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            যদি (pool_addr == 0) {
                ACL_CHECK(aclrtReserveMemAddress(&pool_addr, max_size, 0, রিক্ত, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((বর্ণ *) pool_addr + pool_size, reserve_size, 0, handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((বর্ণ *) pool_addr + pool_size);

            // add to the pool
            pool_size += reserve_size;

#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO("cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n", device,
                          (unsigned long long) (pool_size / 1024 / 1024),
                          (unsigned long long) (reserve_size / 1024 / 1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        শূন্য * ptr   = (শূন্য *) ((বর্ণ *) pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size,
                      (unsigned long long) ptr);
#endif
        প্রত্যাগম ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    শূন্য মুক্ত(শূন্য * ptr, size_t size) override {
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size,
                      (unsigned long long) ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (শূন্য *) ((বর্ণ *) pool_addr + pool_used));
    }
};

/**
 * @brief Create a new CANN pool for a specific device.
 *
 * Factory method to create a new CANN pool object based on the device type.
 *
 * @param device The device ID for which to create the pool.
 * @return A unique pointer to the created CANN pool.
 */
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(পূর্ণাঙ্ক device) {
    std::string mem_pool_type = get_env_as_lowercase("GGML_CANN_MEM_POOL").value_or("");

    যদি (mem_pool_type == "prio") {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        প্রত্যাগম std::unique_ptr<ggml_cann_pool>(নূতন ggml_cann_pool_buf_prio(device));
    }

    যদি (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        প্রত্যাগম std::unique_ptr<ggml_cann_pool>(নূতন ggml_cann_pool_vmm(device));
    }

    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    প্রত্যাগম std::unique_ptr<ggml_cann_pool>(নূতন ggml_cann_pool_buf(device));
}

// cann buffer

/**
 * @brief Tracks multi-threaded write progress for a single tensor.
 *
 * When multiple threads call set_tensor on different chunks of the same tensor,
 * this tracker accumulates progress and defers post-processing (quantized format
 * transform or ND-to-NZ conversion) until all data has been written.
 */
সংরচনা TensorSetTracker {
    std::mutex mtx;                   ///< Protects concurrent access to this tracker
    size_t bytes_written = 0;         ///< Accumulated bytes written so far
    size_t total_bytes = 0;           ///< Target size (full tensor)
    std::vector<uint8_t> host_buffer; ///< Host staging buffer for quantized tensors
};

/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
সংরচনা ggml_backend_cann_buffer_context {
    int32_t device;             ///< The device ID associated with this buffer context.
    শূন্য *  dev_ptr = nullptr;  ///< Pointer to the device memory allocated for the buffer.

    std::mutex tracker_mutex;   ///< Protects the trackers map
    std::unordered_map<শূন্য *, std::unique_ptr<TensorSetTracker>> trackers;

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_buffer_context(int32_t device, শূন্য * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_buffer_context() { ACL_CHECK(aclrtFree(dev_ptr)); }

    /**
     * @brief Get or create a tracker for the given tensor.
     */
    TensorSetTracker * get_or_create_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        auto চাবি = tensor->data;
        auto it = trackers.find(চাবি);
        যদি (it == trackers.end()) {
            auto tracker = std::make_unique<TensorSetTracker>();
            tracker->total_bytes = ggml_nbytes(tensor);
            auto * ptr = tracker.get();
            trackers[চাবি] = std::move(tracker);
            প্রত্যাগম ptr;
        }
        প্রত্যাগম it->second.get();
    }

    /**
     * @brief Remove the tracker for the given tensor.
     */
    শূন্য remove_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        trackers.erase(tensor->data);
    }
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
সংরচনা ggml_backend_cann_buffer_type_context {
    int32_t     device; /**< Device identifier associated with the buffer context. */
    std::string name;   /**< Name associated with the buffer context. */
};

/**
 * @brief Retrieves the name associated with a CANN buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * CANN buffer type context.
 *
 * @param buft Pointer to the buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
স্থির ধ্রুব বর্ণ * ggml_backend_cann_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cann_buffer_type_context * buft_ctx = (ggml_backend_cann_buffer_type_context *) buft->context;

    প্রত্যাগম buft_ctx->name.c_str();
}

/**
 * @brief Checks if the backend buffer type is associated with the CANN backend.
 *
 * This function checks whether the provided backend buffer type is associated
 * with the CANN backend based on the comparison of its name retrieval function
 * pointer.
 *
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the buffer type is associated with the CANN
 * backend, otherwise false.
 */
স্থির তার্কিক ggml_backend_buft_is_cann(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম buft->iface.get_name == ggml_backend_cann_buffer_type_name;
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
স্থির শূন্য ggml_backend_cann_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;
    delete ctx;
}

/**
 * @brief Retrieve the base pointer of a CANN buffer.
 *
 * This function returns the base pointer of a CANN buffer, which points to the
 * device memory allocated for the buffer.
 *
 * @param buffer The CANN buffer whose base pointer is to be retrieved.
 * @return A pointer to the base of the device memory allocated for the buffer.
 */
স্থির শূন্য * ggml_backend_cann_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;
    প্রত্যাগম ctx->dev_ptr;
}

/**
 * @brief Transform quantized Q4.0 tensor data into a format suitable for CANN
 * processing.
 *
 * This function transforms quantized Q4.0 tensor data into a format suitable
 * for CANN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by CANN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data in Q4.0 format.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
স্থির শূন্য ggml_backend_cann_transform_q4_0(ggml_tensor * tensor, ধ্রুব শূন্য * src, শূন্য * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_bytes = n_elems * পরিমাণ(uint8_t) / 2;

    uint8_t *  quant_offset = (uint8_t *) dst;
    uint16_t * scale_offset = (uint16_t *) ((বর্ণ *) dst + quant_bytes);

    প্রতি (পূর্ণাঙ্ক i = 0; i < groups; i++) {
        ধ্রুব block_q4_0 * group = (ধ্রুব block_q4_0 *) ((ধ্রুব বর্ণ *) src + i * পরিমাণ(block_q4_0));
        *scale_offset            = group->d;
        scale_offset++;

        // 0-15
        প্রতি (পূর্ণাঙ্ক j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] & 0x0F);
            (*quant_offset) |= ((group->qs[j + 1] << 4));
            quant_offset++;
        }

        // 16-31
        প্রতি (পূর্ণাঙ্ক j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] >> 4);
            (*quant_offset) |= (group->qs[j + 1] & 0xF0);
            quant_offset++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    প্রতি (quant_offset = (uint8_t *) dst; quant_offset < (uint8_t *) dst + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
}

/**
 * @brief Transform CANN processed data back into quantized Q4.0 format.
 *
 * This function transforms CANN processed data back into quantized Q4.0 format.
 * It reverses the transformation performed by
 * ggml_backend_cann_transform_q4_0(), converting the data back into its
 * original quantized form.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source buffer containing transformed data.
 * @param dst Pointer to the destination buffer where the Q4.0 formatted data
 * will be stored.
 */
স্থির শূন্য ggml_backend_cann_transform_back_q4_0(ধ্রুব ggml_tensor * tensor, শূন্য * src, শূন্য * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_bytes = n_elems * পরিমাণ(uint8_t) / 2;

    uint8_t *  quant_offset = (uint8_t *) src;
    uint16_t * scale_offset = (uint16_t *) ((বর্ণ *) src + quant_bytes);

    প্রতি (; quant_offset < (uint8_t *) src + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
    quant_offset = (uint8_t *) src;

    প্রতি (পূর্ণাঙ্ক i = 0; i < groups; i++) {
        block_q4_0 * group = (block_q4_0 *) ((বর্ণ *) dst + i * পরিমাণ(block_q4_0));
        group->d           = *scale_offset;
        scale_offset++;

        // 0-15
        প্রতি (পূর্ণাঙ্ক j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j]     = ((*quant_offset) & 0x0F);
            group->qs[j + 1] = ((*quant_offset) >> 4);
            quant_offset++;
        }

        // 16-31
        প্রতি (পূর্ণাঙ্ক j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j] |= ((*quant_offset) << 4);
            group->qs[j + 1] |= ((*quant_offset) & 0xF0);
            quant_offset++;
        }
    }
}

/**
 * @brief Transform quantized Q8.0 tensor data into a format suitable for CANN
 * processing.
 *
 * This function transforms quantized Q8.0 tensor data into a format suitable
 * for CANN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by CANN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data in Q8.0 format.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
স্থির শূন্য ggml_backend_cann_transform_q8_0(ggml_tensor * tensor, ধ্রুব শূন্য * src, শূন্য * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_bytes = n_elems * পরিমাণ(uint8_t);

    uint8_t *  quant_offset = (uint8_t *) dst;
    uint16_t * scale_offset = (uint16_t *) ((বর্ণ *) dst + quant_bytes);

    প্রতি (পূর্ণাঙ্ক i = 0; i < groups; i++) {
        ধ্রুব block_q8_0 * group = (ধ্রুব block_q8_0 *) ((ধ্রুব বর্ণ *) src + i * পরিমাণ(block_q8_0));
        *scale_offset            = group->d;
        scale_offset++;
        size_t group_quant_size = QK8_0 * পরিমাণ(uint8_t);
        memcpy(quant_offset, group->qs, group_quant_size);
        quant_offset += group_quant_size;
    }
}

/**
 * @brief Transform CANN processed data back into quantized Q8.0 format.
 *
 * This function transforms CANN processed data back into quantized Q8.0 format.
 * It reverses the transformation performed by
 * ggml_backend_cann_transform_q8_0(), converting the data back into its
 * original quantized form.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source buffer containing transformed data.
 * @param dst Pointer to the destination buffer where the Q8.0 formatted data
 * will be stored.
 */
স্থির শূন্য ggml_backend_cann_transform_back_q8_0(ধ্রুব ggml_tensor * tensor, ধ্রুব শূন্য * src, শূন্য * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_bytes = n_elems * পরিমাণ(uint8_t);

    ধ্রুব uint8_t *  quant_offset = (ধ্রুব uint8_t *) src;
    ধ্রুব uint16_t * scale_offset = (ধ্রুব uint16_t *) ((ধ্রুব বর্ণ *) src + quant_bytes);

    প্রতি (পূর্ণাঙ্ক i = 0; i < groups; i++) {
        block_q8_0 * group = (block_q8_0 *) ((বর্ণ *) dst + i * পরিমাণ(block_q8_0));
        group->d           = *scale_offset;
        scale_offset++;
        size_t group_quant_size = QK8_0 * পরিমাণ(uint8_t);
        memcpy(group->qs, quant_offset, group_quant_size);
        quant_offset += group_quant_size;
    }
}

/**
 * @brief Transform tensor data based on its type for CANN processing.
 *
 * This function transforms tensor data based on its quantization type for CANN
 * processing. It dispatches the transformation based on the tensor's type to
 * specialized functions handling Q4.0 and Q8.0 formats.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data to be transformed.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
স্থির শূন্য ggml_backend_cann_transform(ggml_tensor * tensor, ধ্রুব শূন্য * src, শূন্য * dst) {
    বিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            ভঙ্গ;
        স্বভাব:
            ভঙ্গ;
    }
}

/**
 * @brief Transform CANN processed data back into tensor data based on its type.
 *
 * This function transforms CANN processed data back into tensor data based on
 * its quantization type for Q4.0 and Q8.0 formats. It dispatches the
 * transformation based on the tensor's type to specialized functions.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data containing CANN processed data.
 * @param dst Pointer to the destination buffer where transformed tensor data
 * will be stored.
 */
স্থির শূন্য ggml_backend_cann_transform_back(ধ্রুব ggml_tensor * tensor, শূন্য * src, শূন্য * dst) {
    বিকল্প (tensor->type) {
        পক্ষ GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_back_q4_0(tensor, src, dst);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_back_q8_0(tensor, src, dst);
            ভঙ্গ;
        স্বভাব:
            ভঙ্গ;
    }
}

/**
 * @brief Check if transformation is needed for a given tensor type.
 *
 * This function checks if transformation is needed for a given tensor type
 * to prepare data for CANN processing.
 *
 * @param type The tensor type to check.
 * @return true if transformation is needed, false otherwise.
 */
স্থির তার্কিক need_transform(ggml_type type) {
    বিকল্প (type) {
        পক্ষ GGML_TYPE_Q4_0:
        পক্ষ GGML_TYPE_Q8_0:
            প্রত্যাগম সত্য;
        স্বভাব:
            প্রত্যাগম অসত্য;
    }
}

/**
 * @brief Initialize a tensor using data from a CANN buffer.
 *
 * This function initializes a tensor using data from a CANN buffer.
 * It handles special cases such as views and quantization.
 *
 * @param buffer The CANN buffer from which to initialize the tensor.
 * @param tensor Pointer to the tensor to be initialized.
 */
স্থির গণনা ggml_status ggml_backend_cann_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    যদি (tensor->view_src != রিক্ত && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        প্রত্যাগম GGML_STATUS_SUCCESS;
    }

    // TODO: cann backend doesn't support quantized yet. Just leave the code
    // here.
    যদি (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size   = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        যদি (padded_size > original_size && tensor->view_src == nullptr) {
            size_t memset_size = padded_size - original_size;
            ACL_CHECK(aclrtMemset((বর্ণ *) tensor->data + original_size, memset_size, 0, memset_size));
        }
    }
    প্রত্যাগম GGML_STATUS_SUCCESS;
}

/**
 * @brief Workspace for caching NZ buffers per device.
 *
 * This struct manages a device buffer used in NZ computations. It supports
 * allocation, reallocation, and clearing of cached memory. The struct is
 * designed to be used with a global array, one per device.
 */
সংরচনা ggml_cann_nz_workspace {
    std::mutex mtx;    // Protects ptr/allocated from concurrent access
    শূন্য * ptr;        // Pointer to allocated device buffer
    size_t allocated;  // Size of currently allocated buffer in bytes

    /**
     * @brief Constructor. Initializes the workspace with no allocated memory.
     */
    ggml_cann_nz_workspace() : ptr(nullptr), allocated(0) {}

    /**
     * @brief Free cached memory and reset the workspace.
     *
     * If a buffer has been allocated, this function releases it using
     * aclrtFree and resets internal state.
     */
    শূন্য clear() {
        যদি (ptr) {
            ACL_CHECK(aclrtFree(ptr));
            ptr       = nullptr;
            allocated = 0;
        }
    }

    /**
     * @brief Allocate or reallocate the workspace buffer.
     *
     * If the requested size is larger than the currently allocated size,
     * the old buffer will be freed and a new buffer of the requested size
     * will be allocated on the device.
     *
     * @param new_size Size in bytes to allocate for the workspace.
     */
    শূন্য realloc(size_t new_size) {
        যদি (new_size > allocated) {
            clear();
            ACL_CHECK(aclrtMalloc(&ptr, new_size, ACL_MEM_MALLOC_HUGE_FIRST));
            allocated = new_size;
        }
    }

    /**
     * @brief Get the device buffer pointer.
     *
     * @return Pointer to the allocated buffer, or nullptr if not allocated.
     */
    শূন্য * get() ধ্রুব { প্রত্যাগম ptr; }
};

/**
 * @brief Global array of NZ workspaces, one per device.
 */
স্থির ggml_cann_nz_workspace g_nz_workspaces[GGML_CANN_MAX_DEVICES];

/**
 * @brief Convert tensor weights to NZ format using Ascend CANN API.
 *
 * This function creates a transposed tensor descriptor and performs the
 * TransMatmulWeight operation. Converting tensor formats can significantly
 * improve performance on certain hardware.
 *
 * @param tensor Pointer to the input ggml_tensor containing the weights.
 * @param offset Byte offset within the tensor data buffer where weights start.
 * @param device device id.
 *
 * @note The workspace buffer used in this function is managed globally and reused
 *       across calls. This reduces overhead from repeated memory allocation and deallocation.
 */
স্থির শূন্য weight_format_to_nz(ggml_tensor * tensor, পূর্ণাঙ্ক device) {
    acl_tensor_ptr weightTransposed = ggml_cann_create_tensor(tensor, tensor->ne, tensor->nb, 2, ACL_FORMAT_ND, 0);
    uint64_t       workspaceSize    = 0;
    aclOpExecutor * executor;

    // TransMatmulWeight
    ACL_CHECK(aclnnTransMatmulWeightGetWorkspaceSize(weightTransposed.get(), &workspaceSize, &executor));

    std::lock_guard<std::mutex> lock(g_nz_workspaces[device].mtx);
    // Avoid frequent malloc/free of the workspace.
    g_nz_workspaces[device].realloc(workspaceSize);

    শূন্য * g_nz_workspace = g_nz_workspaces[device].get();

    ACL_CHECK(aclnnTransMatmulWeight(g_nz_workspace, workspaceSize, executor, nullptr));
}

// TODO: need handle tensor which has paddings.
/**
 * @brief Set tensor data in a CANN buffer.
 *
 * This function sets tensor data in a CANN buffer, handling transformations
 * if needed based on the tensor's type. It supports multi-threaded calls
 * where different threads write different chunks of the same tensor.
 *
 * For quantized tensors (Q4_0/Q8_0), data is staged in a host buffer and
 * the format transform is deferred until all chunks are written.
 * For NZ weight tensors, chunks are uploaded directly but the ND-to-NZ
 * conversion is deferred until all chunks are written.
 *
 * @param buffer The CANN buffer where the tensor data will be set.
 * @param tensor Pointer to the tensor whose data will be set.
 * @param data Pointer to the source data to be copied into the tensor.
 * @param offset Offset in the source data from where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
স্থির শূন্য ggml_backend_cann_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *         tensor,
                                                ধ্রুব শূন্য *          data,
                                                size_t                offset,
                                                size_t                size) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;

    ggml_cann_set_device(ctx->device);

    // Only check env once.
    স্থির তার্কিক weight_to_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    তার্কিক is_quantized = need_transform(tensor->type);
    তার্কিক is_nz        = !is_quantized && tensor->type != GGML_TYPE_BF16 && weight_to_nz &&
                 is_matmul_weight((ধ্রুব ggml_tensor *) tensor);

    // Plain tensor (not quantized, not NZ): direct copy, no tracking needed
    যদি (!is_quantized && !is_nz) {
        ACL_CHECK(aclrtMemcpy((বর্ণ *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
        প্রত্যাগম;
    }

    // Single-shot write (full tensor at once): handle directly without tracking overhead
    যদি (offset == 0 && size == ggml_nbytes(tensor)) {
        যদি (is_quantized) {
            শূন্য * transform_buffer = malloc(size);
            ggml_backend_cann_transform(tensor, data, transform_buffer);
            ACL_CHECK(aclrtMemcpy(tensor->data, size, transform_buffer, size, ACL_MEMCPY_HOST_TO_DEVICE));
            মুক্ত(transform_buffer);
        } অন্যথা {
            // NZ weight
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ACL_CHECK(aclrtMemcpy(tensor->data, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
            weight_format_to_nz(tensor, ctx->device);
        }
        প্রত্যাগম;
    }

    // Chunked write: use tracker to accumulate progress and defer transform/conversion
    TensorSetTracker * tracker = ctx->get_or_create_tracker(tensor);
    std::unique_lock<std::mutex> lock(tracker->mtx);

    যদি (is_quantized) {
        // Stage data in host buffer; transform requires full tensor data
        যদি (tracker->host_buffer.empty()) {
            tracker->host_buffer.resize(tracker->total_bytes);
        }
        memcpy(tracker->host_buffer.data() + offset, data, size);
    } অন্যথা {
        // NZ weight: upload chunk to device immediately, defer conversion
        ACL_CHECK(aclrtMemcpy((বর্ণ *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    tracker->bytes_written += size;

    // All chunks received: perform deferred transform/conversion
    যদি (tracker->bytes_written >= tracker->total_bytes) {
        যদি (is_quantized) {
            শূন্য * transform_buffer = malloc(tracker->total_bytes);
            ggml_backend_cann_transform(tensor, tracker->host_buffer.data(), transform_buffer);
            ACL_CHECK(aclrtMemcpy(tensor->data, tracker->total_bytes, transform_buffer, tracker->total_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            মুক্ত(transform_buffer);
        }

        যদি (is_nz) {
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            weight_format_to_nz(tensor, ctx->device);
        }

        // Unlock before removing tracker, as remove_tracker destroys the mutex
        lock.unlock();
        ctx->remove_tracker(tensor);
    }
}

/**
 * @brief Get tensor data from a CANN buffer.
 *
 * This function retrieves tensor data from a CANN buffer, handling
 * transformations if needed based on the tensor's type.
 *
 * @param buffer The CANN buffer from which to retrieve tensor data.
 * @param tensor Pointer to the tensor whose data will be retrieved.
 * @param data Pointer to the destination buffer where the tensor data will be
 * copied.
 * @param offset Offset in the destination buffer where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
স্থির শূন্য ggml_backend_cann_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                ধ্রুব ggml_tensor *   tensor,
                                                শূন্য *                data,
                                                size_t                offset,
                                                size_t                size) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;

    ggml_cann_set_device(ctx->device);

    যদি (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(data, size, (বর্ণ *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
    } অন্যথা {
        শূন্য * transform_buffer = malloc(size);
        ACL_CHECK(aclrtMemcpy(transform_buffer, size, (বর্ণ *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
        ggml_backend_cann_transform_back(tensor, transform_buffer, data);
        মুক্ত(transform_buffer);
    }
}

/**
 * @brief Copy tensor data between CANN buffers if possible.
 *
 * This function copies tensor data between CANN buffers if the source and
 * destination buffers are CANN buffers and they meet the necessary conditions
 * (same device or devices can access each other).
 *
 * @param buffer The destination CANN buffer where the tensor data will be
 * copied.
 * @param src Pointer to the source tensor whose data will be copied.
 * @param dst Pointer to the destination tensor where the data will be copied.
 * @return true if the copy operation succeeded, false otherwise.
 */
স্থির তার্কিক ggml_backend_cann_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                ধ্রুব ggml_tensor *   src,
                                                ggml_tensor *         dst) {
    যদি (ggml_backend_buft_is_cann(src->buffer->buft)) {
        ggml_backend_cann_buffer_context * src_ctx = (ggml_backend_cann_buffer_context *) src->buffer->context;
        ggml_backend_cann_buffer_context * dst_ctx = (ggml_backend_cann_buffer_context *) buffer->context;

        size_t memcpy_size = ggml_nbytes(src);
        // Same device.
        যদি (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((বর্ণ *) dst->data, memcpy_size, (ধ্রুব বর্ণ *) src->data, memcpy_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE));
            প্রত্যাগম সত্য;
        } অন্যথা {
#ifdef ASCEND_310P
            // TODO: Support 310p P2P copy
            প্রত্যাগম অসত্য;
#endif
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device, dst_ctx->device));
            যদি (canAccessPeer) {
                ggml_cann_set_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((বর্ণ *) dst->data, memcpy_size, (ধ্রুব বর্ণ *) src->data, memcpy_size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE));
                প্রত্যাগম সত্য;
            }
        }
    }
    প্রত্যাগম অসত্য;
}

/**
 * @brief Set a region of a tensor's device memory to a specified value.
 *
 * @param buffer The CANN buffer containing the tensor.
 * @param tensor Pointer to the tensor whose memory will be set.
 * @param value The value to which each byte in the region will be set.
 * @param offset Byte offset within the tensor's data to start setting.
 * @param size Number of bytes to set.
 */
স্থির শূন্য ggml_backend_cann_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset((বর্ণ *) tensor->data + offset, size, value, size));
}

/**
 * @brief Clear a CANN buffer by setting all its memory to a specified value.
 *
 * This function clears a CANN buffer by setting all its memory to a specified
 * value.
 *
 * @param buffer The CANN buffer to be cleared.
 * @param value The value to which each byte in the buffer will be set.
 */
স্থির শূন্য ggml_backend_cann_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
স্থির ধ্রুব ggml_backend_buffer_i ggml_backend_cann_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cann_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cann_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cann_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cann_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cann_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cann_buffer_get_tensor,
    /* .set_tensor_2d   = */ রিক্ত,
    /* .get_tensor_2d   = */ রিক্ত,
    /* .cpy_tensor      = */ ggml_backend_cann_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_buffer_clear,
    /* .reset           = */ রিক্ত,
};

/**
 * @brief Allocates a new CANN buffer of the specified type and size.
 *
 * This function allocates a new CANN buffer on the specified device with the
 * given size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
স্থির ggml_backend_buffer_t ggml_backend_cann_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cann_buffer_type_context * buft_ctx = (ggml_backend_cann_buffer_type_context *) buft->context;

    ggml_cann_set_device(buft_ctx->device);

    ধ্রুব size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    যদি (size == 0) {
        size = alignment;
    }
    শূন্য *   dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    যদি (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, buft_ctx->device, acl_Get_Recent_Err_বার্তা());
        প্রত্যাগম nullptr;
    }

    ggml_backend_cann_buffer_context * ctx = নূতন ggml_backend_cann_buffer_context(buft_ctx->device, dev_ptr);

    প্রত্যাগম ggml_backend_buffer_init(buft, ggml_backend_cann_buffer_interface, ctx, size);
}

/**
 * @brief Retrieves the memory alignment requirement for CANN buffers of this
 * type.
 *
 * This function returns the alignment requirement in bytes for memory allocated
 * by the CANN buffer type.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @return The alignment requirement in bytes (fixed at 128 bytes for CANN
 * buffers).
 */
স্থির size_t ggml_backend_cann_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম 128;

    GGML_UNUSED(buft);
}

/**
 * @brief Calculates the allocation size required for a tensor in a CANN buffer.
 *
 * Computes the total allocation size needed for storing the tensor's data in a
 * CANN buffer, considering any necessary padding or adjustments for quantized
 * types.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @param tensor Pointer to the tensor for which the allocation size is
 * calculated.
 * @return The total allocation size in bytes required for the tensor in the
 * CANN buffer.
 */
স্থির size_t ggml_backend_cann_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                           ধ্রুব ggml_tensor *        tensor) {
    size_t  size = ggml_nbytes(tensor);
    int64_t ne0  = tensor->ne[0];

    // Only check env once.
    স্থির তার্কিক weight_to_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);
    যদি (ggml_is_quantized(tensor->type)) {
        যদি (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    } অন্যথা যদি (weight_to_nz && tensor->type != GGML_TYPE_BF16
               && is_matmul_weight((ধ্রুব ggml_tensor *) tensor)) {
        // NZ format weight are not support quantized yet.
        // If ND tensor transform to NZ, size may changed.
        int64_t shape[] = { tensor->ne[1], tensor->ne[0] };
        GGML_ASSERT(tensor->ne[2] == 1);
        GGML_ASSERT(tensor->ne[3] == 1);
        ধ্রুব aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
        size_t              new_size;
        ACL_CHECK(aclnnCalculateMatmulWeightSizeV2(acl_shape, ggml_cann_type_mapping(tensor->type), &new_size));
        ACL_CHECK(aclDestroyIntArray(acl_shape));
        size = std::max(size, new_size);
    }

    প্রত্যাগম size;

    GGML_UNUSED(buft);
}

স্থির তার্কিক ggml_backend_cann_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম অসত্য;

    GGML_UNUSED(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
স্থির ধ্রুব ggml_backend_buffer_type_i ggml_backend_cann_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cann_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_cann_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ রিক্ত,
    /* .get_alignment       = */ ggml_backend_cann_buffer_type_get_alignment,
    /* .get_max_size        = */ রিক্ত,  // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cann_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ রিক্ত,
    /* .is_host             = */ ggml_backend_cann_buffer_type_is_host,
};

/**
 * @brief Retrieves the CANN buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
ggml_backend_buffer_type_t ggml_backend_cann_buffer_type(int32_t device) {
    স্থির std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    যদি (device >= ggml_backend_cann_get_device_গুনতি()) {
        প্রত্যাগম nullptr;
    }

    স্থির ggml_backend_buffer_type ggml_backend_cann_buffer_types[GGML_CANN_MAX_DEVICES];

    স্থির তার্কিক ggml_backend_cann_buffer_type_initialized = অসত্য;

    যদি (!ggml_backend_cann_buffer_type_initialized) {
        প্রতি (int32_t i = 0; i < ggml_cann_info().device_গুনতি; i++) {
            ggml_backend_cann_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cann_buffer_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), i),
                /* .context  = */
                নূতন ggml_backend_cann_buffer_type_context{ i, "CANN" + std::to_string(i) },
            };
        }
        ggml_backend_cann_buffer_type_initialized = সত্য;
    }

    প্রত্যাগম &ggml_backend_cann_buffer_types[device];
}

/**
 * @brief Retrieves the name associated with a CANN host buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * CANN host buffer type context.
 *
 * @param buft Pointer to the host buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
স্থির ধ্রুব বর্ণ * ggml_backend_cann_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    প্রত্যাগম "CANN_Host";

    GGML_UNUSED(buft);
}

/**
 * @brief Retrieves the name associated with a CANN host buffer.
 *
 * This function returns the descriptive name associated with the specified
 * CANN host buffer context.
 *
 * @param buft Pointer to the host buffer context.
 * @return Const pointer to the C-style string containing the name.
 */
স্থির ধ্রুব বর্ণ * ggml_backend_cann_host_buffer_name(ggml_backend_buffer_t buffer) {
    প্রত্যাগম "CANN_Host";

    GGML_UNUSED(buffer);
}

/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer, including
 * its context.
 *
 * @param buffer The CANN host buffer to free.
 */
স্থির শূন্য ggml_backend_cann_host_buffer_free(ggml_backend_buffer_t buffer) {
    ACL_CHECK(aclrtFreeHost(buffer->context));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
স্থির শূন্য * ggml_cann_host_malloc(size_t size) {
    যদি (getenv("GGML_CANN_NO_PINNED") != nullptr) {
        প্রত্যাগম nullptr;
    }

    ধ্রুব size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    যদি (size == 0) {
        size = alignment;
    }

    শূন্য *   hostPtr = nullptr;
    aclError err     = aclrtMallocHost((শূন্য **) &hostPtr, size);
    যদি (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                      acl_Get_Recent_Err_বার্তা());
        প্রত্যাগম nullptr;
    }
    প্রত্যাগম hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
স্থির ggml_backend_buffer_t ggml_backend_cann_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                             size_t                     size) {
    শূন্য * hostPtr = ggml_cann_host_malloc(size);

    যদি (hostPtr == nullptr) {
        // fallback to cpu buffer
        প্রত্যাগম ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(hostPtr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cann_host_buffer_free;

    প্রত্যাগম buffer;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_buffer_type_t ggml_backend_cann_host_buffer_type() {
    স্থির সংরচনা ggml_backend_buffer_type ggml_backend_cann_buffer_type_host = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cann_host_buffer_type_name,
                           /* .alloc_buffer         = */ ggml_backend_cann_host_buffer_type_alloc_buffer,
                           /* .alloc_buffer_n       = */ রিক্ত,
                           /* .get_alignment        = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                           /* .get_max_size         = */ রিক্ত,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                           /* .get_alloc_size_n     = */ রিক্ত,
                           /* .is_host              = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
        /* .context  = */ nullptr,
    };

    প্রত্যাগম &ggml_backend_cann_buffer_type_host;
}

/**
 * @brief Computes the forward operation for a given tensor using CANN
 * operations.
 *
 * This function selects the appropriate CANN operation based on the type of
 * operation specified in the tensor and performs the computation.
 *
 * @param ctx The CANN context containing necessary resources and
 * configurations.
 * @param dst The destination tensor where the result of the computation will be
 * stored.
 * @return true if the computation was successful; false otherwise.
 */
স্থির তার্কিক ggml_cann_compute_forward(ggml_backend_cann_context & ctx, সংরচনা ggml_tensor * dst) {
    বিকল্প (dst->op) {
        পক্ষ GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_GET_ROWS:
            ggml_cann_get_rows(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SET_ROWS:
            ggml_cann_set_rows(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_ADD:
        পক্ষ GGML_OP_ADD1:
            ggml_cann_দ্বিমিক_op<aclnn_add>(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SUB:
            ggml_cann_দ্বিমিক_op<aclnn_sub>(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_MUL:
            ggml_cann_দ্বিমিক_op<aclnn_mul>(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_DIV:
            ggml_cann_দ্বিমিক_op<aclnn_div>(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_UNARY:
            বিকল্প (ggml_get_unary_op(dst)) {
                পক্ষ GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_OP_UNARY(Abs);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_OP_UNARY(Neg);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_GELU:
                পক্ষ GGML_UNARY_OP_GELU_ERF:
                    // aclnnGelu internally uses the erf-based approximation.
                    GGML_CANN_CALL_OP_UNARY(Gelu);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_OP_UNARY(Silu);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_GELU_QUICK:
                    {
                        auto lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                        };
                        ggml_cann_op_unary(lambda, ctx, dst);
                    }
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_TANH:
                    GGML_CANN_CALL_OP_UNARY(Tanh);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_RELU:
                    GGML_CANN_CALL_OP_UNARY(Relu);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_SIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Sigmoid);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_HARDSIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Hardsigmoid);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_HARDSWISH:
                    GGML_CANN_CALL_OP_UNARY(Hardswish);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_EXP:
                    GGML_CANN_CALL_OP_UNARY(Exp);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_ELU:
                    ggml_cann_elu(ctx, dst);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_SGN:
                    GGML_CANN_CALL_OP_UNARY(Sign);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_STEP:
                    ggml_cann_step(ctx, dst);
                    ভঙ্গ;
                পক্ষ GGML_UNARY_OP_SOFTPLUS:
                    ggml_cann_softplus(ctx, dst);
                    ভঙ্গ;
                স্বভাব:
                    প্রত্যাগম অসত্য;
            }
            ভঙ্গ;
        পক্ষ GGML_OP_GLU:
            বিকল্প (ggml_get_glu_op(dst)) {
                পক্ষ GGML_GLU_OP_REGLU:
                    GGML_CANN_CALL_OP_UNARY_GATED(Relu);
                    ভঙ্গ;
                পক্ষ GGML_GLU_OP_GEGLU:
                    ggml_cann_geglu(ctx, dst, 0);  // approximate=0 → tanh
                    ভঙ্গ;
                পক্ষ GGML_GLU_OP_GEGLU_ERF:
                    ggml_cann_geglu(ctx, dst, 1);  // approximate=1 → erf
                    ভঙ্গ;
                পক্ষ GGML_GLU_OP_SWIGLU:
                    ggml_cann_swiglu(ctx, dst);
                    ভঙ্গ;
                পক্ষ GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cann_swiglu_clamp(ctx, dst);
                    ভঙ্গ;
                পক্ষ GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cann_geglu_quick(ctx, dst);
                    ভঙ্গ;
                স্বভাব:
                    প্রত্যাগম অসত্য;
            }
            ভঙ্গ;
        পক্ষ GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_GROUP_NORM:
            ggml_cann_group_norm(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_L2_NORM:
            ggml_cann_l2_norm(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cann_cross_entropy_loss(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == nullptr);
            dst->src[1] = dst->src[0];
            ggml_cann_দ্বিমিক_op<aclnn_mul>(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SQRT:
            GGML_CANN_CALL_OP_UNARY(Sqrt);
            ভঙ্গ;
        পক্ষ GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SET:
            ggml_cann_set(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_NONE:
        পক্ষ GGML_OP_RESHAPE:
        পক্ষ GGML_OP_VIEW:
        পক্ষ GGML_OP_PERMUTE:
        পক্ষ GGML_OP_TRANSPOSE:
            ভঙ্গ;
        পক্ষ GGML_OP_DIAG_MASK_INF:
            ggml_cann_diag_mask(ctx, dst, -INFINITY);
            ভঙ্গ;
        পক্ষ GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_IM2COL:
            ggml_cann_im2col(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SUM:
            ggml_cann_sum(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SUM_ROWS:
            ggml_cann_sum_rows(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_ARGMAX:
            ggml_cann_argmax(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_COS:
            ggml_cann_op_unary<aclnn_cos>(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SIN:
            ggml_cann_op_unary<aclnn_sin>(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cann_conv_transpose_1d(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_LOG:
            GGML_CANN_CALL_OP_UNARY(Log);
            ভঙ্গ;
        পক্ষ GGML_OP_MEAN:
            ggml_cann_mean(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_PAD_REFLECT_1D:
            ggml_cann_pad_reflect_1d(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_গুনতি_EQUAL:
            ggml_cann_গুনতি_equal(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_OUT_PROD:
            ggml_cann_out_prod(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_GATED_LINEAR_ATTN:
            ggml_cann_gated_linear_attn(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SSM_CONV:
            ggml_cann_ssm_conv(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_CUMSUM:
            ggml_cann_cumsum(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_TRI:
            ggml_cann_tri(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_FILL:
            ggml_cann_fill(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_DIAG:
            ggml_cann_diag(ctx, dst);
            ভঙ্গ;
        পক্ষ GGML_OP_SOLVE_TRI:
            ggml_cann_solve_tri(ctx, dst);
            ভঙ্গ;
        স্বভাব:
            প্রত্যাগম অসত্য;
    }

    প্রত্যাগম সত্য;
}

// backend
/**
 * @brief Retrieves the name associated with the CANN backend.
 *
 * This function returns the name assigned to the CANN backend, which is stored
 * in the context of the provided backend structure.
 *
 * @param backend Pointer to the CANN backend structure.
 * @return A pointer to a constant string representing the backend name.
 */
স্থির ধ্রুব বর্ণ * ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;

    প্রত্যাগম cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
স্থির শূন্য ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    delete cann_ctx;
    delete backend;
}

/**
 * @brief Sets tensor data asynchronously in the CANN backend.
 *
 * This function asynchronously sets tensor data in the CANN backend.
 *
 * @param backend Pointer to the CANN backend structure.
 * @param tensor Pointer to the tensor structure to set data for.
 * @param data Pointer to the host data to copy to the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
স্থির শূন্য ggml_backend_cann_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *  tensor,
                                               ধ্রুব শূন্য *   data,
                                               size_t         offset,
                                               size_t         size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_buffer_t       buf      = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cann_buffer_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync((বর্ণ *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE,
                               cann_ctx->stream()));
}

/**
 * @brief Gets tensor data asynchronously in the CANN backend.
 *
 * This function asynchronously gets tensor data in the CANN backend.
 *
 * @param backend Pointer to the CANN backend structure.
 * @param tensor Pointer to the tensor structure to get data from.
 * @param data Pointer to the host data to copy from the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
স্থির শূন্য ggml_backend_cann_get_tensor_async(ggml_backend_t      backend,
                                               ধ্রুব ggml_tensor * tensor,
                                               শূন্য *              data,
                                               size_t              offset,
                                               size_t              size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_buffer_t       buf      = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cann_buffer_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync(data, size, (বর্ণ *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST,
                               cann_ctx->stream()));
}

/**
 * @brief Asynchronously copies tensor data between CANN backends.
 *
 * This function copies tensor data asynchronously between two CANN backends. It
 * checks if both tensors reside in CANN buffers and whether the devices support
 * peer-to-peer access for direct copying. If not, it returns false.
 *
 * @param backend_src Pointer to the source CANN backend structure.
 * @param backend_dst Pointer to the destination CANN backend structure.
 * @param src Pointer to the source tensor to copy data from.
 * @param dst Pointer to the destination tensor to copy data to.
 * @return true if the copy operation succeeds, false otherwise.
 */
স্থির তার্কিক ggml_backend_cann_cpy_tensor_async(ggml_backend_t      backend_src,
                                               ggml_backend_t      backend_dst,
                                               ধ্রুব ggml_tensor * src,
                                               ggml_tensor *       dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) || ggml_backend_is_cann(backend_dst));

    GGML_ASSERT(!is_matmul_weight((ধ্রুব ggml_tensor *) src));

    যদি (!ggml_backend_buft_is_cann(src->buffer->buft) || !ggml_backend_buft_is_cann(dst->buffer->buft)) {
        প্রত্যাগম অসত্য;
    }

    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    ggml_backend_cann_context * cann_ctx_src = (ggml_backend_cann_context *) backend_src->context;
    ggml_backend_cann_context * cann_ctx_dst = (ggml_backend_cann_context *) backend_dst->context;

    size_t copy_size = ggml_nbytes(dst);
    যদি (copy_size == 0) {
        প্রত্যাগম সত্য;
    }
    যদি (backend_src != backend_dst) {
#ifdef ASCEND_310P
        // TODO: Support 310p P2P copy
        প্রত্যাগম অসত্য;
#endif
        ggml_backend_cann_buffer_context * buf_ctx_src = (ggml_backend_cann_buffer_context *) buf_src->context;
        ggml_backend_cann_buffer_context * buf_ctx_dst = (ggml_backend_cann_buffer_context *) buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device, cann_ctx_dst->device));
        যদি (!canAccessPeer) {
            প্রত্যাগম অসত্য;
        }

        // need open both directions for memcpyasync between devices.
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_set_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        // wait for task_queue empty to keep task order.
        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_src->stream()));
        // record event on src stream after the copy
        // TODO: this event is not effective with acl graph mode, change to use aclrtSynchronizeStream
        // if (!cann_ctx_src->copy_event) {
        //     ACL_CHECK(aclrtCreateEventWithFlag(&cann_ctx_src->copy_event, ACL_EVENT_SYNC));
        // }
        // ACL_CHECK(aclrtRecordEvent(cann_ctx_src->copy_event, cann_ctx_src->stream()));

        // // wait on dst stream for the copy to complete
        // ggml_cann_set_device(cann_ctx_dst->device);
        // ACL_CHECK(aclrtStreamWaitEvent(cann_ctx_dst->stream(), cann_ctx_src->copy_event));
        ACL_CHECK(aclrtSynchronizeStream(cann_ctx_src->stream()));
    } অন্যথা {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    প্রত্যাগম সত্য;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
স্থির শূন্য ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_set_device(cann_ctx->device);
    ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
}

/**
 * @brief Check if CANN backend can fuse the specified operation sequence
 *
 * This function determines whether an operation sequence starting from the specified node
 * can be fused into an optimized operation in the CANN backend. Operation fusion can reduce
 * memory access overhead and improve computational efficiency.
 *
 * @param cgraph Pointer to the computation graph
 * @param node_idx Index of the starting node in the computation graph
 * @param ops Sequence of operation types to check for fusion
 * @return true if the operations can be fused
 * @return false if the operations cannot be fused
 */
স্থির তার্কিক ggml_cann_can_fuse(ধ্রুব সংরচনা ggml_cgraph *          cgraph,
                               পূর্ণাঙ্ক                                 গ্রন্থি_idx,
                               std::initializer_list<গণনা ggml_op> ops) {
    যদি (!ggml_can_fuse(cgraph, গ্রন্থি_idx, ops)) {
        প্রত্যাগম অসত্য;
    }

    // CANN backend supports fusing ADD + RMS_NORM operations
    যদি ((ops.size() == 2) && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_RMS_NORM) {
        ggml_tensor * add_গ্রন্থি = cgraph->nodes[গ্রন্থি_idx];
        // TODO: support broadcast for ADD + RMS_NORM
        যদি (add_গ্রন্থি->src[0]->ne[0] != add_গ্রন্থি->src[1]->ne[0] || add_গ্রন্থি->src[0]->ne[1] != add_গ্রন্থি->src[1]->ne[1] ||
            add_গ্রন্থি->src[0]->ne[2] != add_গ্রন্থি->src[1]->ne[2] || add_গ্রন্থি->src[0]->ne[3] != add_গ্রন্থি->src[1]->ne[3]) {
            প্রত্যাগম অসত্য;
        }
        প্রত্যাগম সত্য;
    }

    প্রত্যাগম অসত্য;
}

/**
 * @brief Evaluate the computation graph and optionally capture or execute it using CANN graph API.
 *
 * If CANN graph execution is enabled and graph capture is required, this function begins
 * graph capture, runs the graph, ends capture, and stores the captured graph.
 *
 * Otherwise, it falls back to op-by-op execution using the CANN compute kernel dispatcher.
 *
 * @param cann_ctx                     The CANN backend context.
 * @param cgraph                       The ggml computation graph.
 * @param use_cann_graph               Whether to use CANN graph execution.
 * @param cann_graph_capture_required  Whether graph capture is needed due to graph changes.
 */
স্থির শূন্য evaluate_and_capture_cann_graph(ggml_backend_cann_context * cann_ctx,
                                            ggml_cgraph *               cgraph,
                                            তার্কিক                        use_cann_graph,
                                            তার্কিক                        cann_graph_capture_required) {
#ifdef USE_ACL_GRAPH
    যদি (use_cann_graph && cann_graph_capture_required) {  // Begin CANN graph capture
        ACL_CHECK(aclmdlRICaptureBegin(cann_ctx->stream(), ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    }
#endif  // USE_ACL_GRAPH
    // Only perform the graph execution if CANN graphs are not enabled, or we are capturing the graph.
    // With the use of CANN graphs, the execution will be performed by the graph launch.
    স্থির তার্কিক opt_fusion = parse_bool(get_env_as_lowercase("GGML_CANN_OPERATOR_FUSION").value_or(""));

    যদি (!use_cann_graph || cann_graph_capture_required) {
        প্রতি (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * গ্রন্থি = cgraph->nodes[i];
            যদি (opt_fusion) {
                যদি (ggml_cann_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM })) {
                    ggml_cann_op_add_rms_norm_fused(*cann_ctx, গ্রন্থি, cgraph->nodes[i + 1]);
                    i++;
                    অনুবর্ত;
                }
            }

            যদি (ggml_is_empty(গ্রন্থি) || গ্রন্থি->op == GGML_OP_RESHAPE || গ্রন্থি->op == GGML_OP_TRANSPOSE ||
                গ্রন্থি->op == GGML_OP_VIEW || গ্রন্থি->op == GGML_OP_PERMUTE || গ্রন্থি->op == GGML_OP_NONE) {
                অনুবর্ত;
            }

            যদি ((গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                অনুবর্ত;
            }

            তার্কিক ok = ggml_cann_compute_forward(*cann_ctx, গ্রন্থি);
            যদি (!ok) {
                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, গ্রন্থি->name, ggml_op_name(গ্রন্থি->op));
            }
            GGML_ASSERT(ok);
        }
    }

#ifdef USE_ACL_GRAPH
    যদি (use_cann_graph) {
        GGML_ASSERT(!cann_ctx->graph_lru_cache.cache_list.empty());
        ggml_cann_graph * matched_graph = cann_ctx->graph_lru_cache.cache_list.front();

        যদি (cann_graph_capture_required) {  // End CANN graph capture
            ACL_CHECK(aclmdlRICaptureEnd(cann_ctx->stream(), &matched_graph->graph));
        }

        // Execute CANN graph
        ACL_CHECK(aclmdlRIExecuteAsync(matched_graph->graph, cann_ctx->stream()));
    }
#endif  // USE_ACL_GRAPH
}

/**
 * @brief Computes a computational graph using a CANN backend.
 *
 * This function computes the operations defined in the computational graph
 * using the specified CANN backend.
 *
 * @param backend Pointer to the CANN backend structure to use for computation.
 * @param cgraph Pointer to the computational graph structure containing nodes
 *               representing operations to be computed.
 * @return enum ggml_status Returns GGML_STATUS_SUCCESS if computation
 *         completes successfully, otherwise an appropriate error status.
 */
স্থির গণনা ggml_status ggml_backend_cann_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_set_device(cann_ctx->device);
    g_nz_workspaces[cann_ctx->device].clear();

    // calculate rope cache for fist layer in current device.
    cann_ctx->rope_cache.cached = অসত্য;

    তার্কিক graph_capture_required = অসত্য;
#ifdef USE_ACL_GRAPH
    তার্কিক use_cann_graph = সত্য;

    স্থির তার্কিক prefill_use_graph = parse_bool(get_env_as_lowercase("GGML_CANN_PREFILL_USE_GRAPH").value_or(""));
    যদি (!prefill_use_graph) {
        // Do not use acl_graph for prefill.
        প্রতি (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * গ্রন্থি = cgraph->nodes[i];
            // TODO: Optimize here. Currently, we can only
            // get seq_len by FA's input.
            যদি (গ্রন্থি->op == GGML_OP_FLASH_ATTN_EXT) {
                // Q -> src[0], shape: [B, S, N, D]
                use_cann_graph = (গ্রন্থি->src[0]->ne[1] == 1);
                ভঙ্গ;
            }
        }
    }

    যদি (!cann_ctx->acl_graph_mode) {
        use_cann_graph = অসত্য;
    }

    যদি (use_cann_graph) {
        // If no matching graph is found, the graph needs to be recaptured.
        graph_capture_required = !cann_ctx->graph_lru_cache.find_and_move_to_front(cgraph);

        যদি (graph_capture_required) {
            // If no matching graph is found, add a new ACL graph.
            ggml_cann_graph * new_graph = ggml_cann_graph::create_from_cgraph(cgraph);
            cann_ctx->graph_lru_cache.ঠেলো(new_graph);

            // Pre-load rope cache before graph capture.  During capture the
            // stream cannot perform host-to-device memcpy or device memory
            // malloc/free.  Running the full cache init now populates the
            // cache metadata so these branches are skipped during capture,
            // while also warming up the memory pool.
            প্রতি (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * গ্রন্থি = cgraph->nodes[i];
                যদি (গ্রন্থি->op == GGML_OP_ROPE) {
                    ggml_cann_rope_cache_preload(*cann_ctx, গ্রন্থি);
                    ভঙ্গ;
                }
            }
        }
    }
#অন্যথা
    তার্কিক use_cann_graph = অসত্য;
#endif  // USE_ACL_GRAPH
    evaluate_and_capture_cann_graph(cann_ctx, cgraph, use_cann_graph, graph_capture_required);

    প্রত্যাগম GGML_STATUS_SUCCESS;
}

/**
 * @brief Checks if the CANN backend supports a specific operation.
 *
 * This function checks whether the specified operation is supported by the
 * CANN backend.
 *
 * @param backend Pointer to the CANN backend structure to check support for
 *                the operation.
 * @param op Pointer to the tensor representing the operation to check.
 * @return bool Returns true if the operation is supported by the backend,
 *              otherwise false.
 */
স্থির তার্কিক ggml_backend_cann_supports_op(ggml_backend_dev_t dev, ধ্রুব ggml_tensor * op) {
    বিকল্প (op->op) {
        পক্ষ GGML_OP_UNARY:
            বিকল্প (ggml_get_unary_op(op)) {
                পক্ষ GGML_UNARY_OP_ABS:
                পক্ষ GGML_UNARY_OP_NEG:
                পক্ষ GGML_UNARY_OP_GELU:
                পক্ষ GGML_UNARY_OP_SILU:
                পক্ষ GGML_UNARY_OP_RELU:
                পক্ষ GGML_UNARY_OP_SIGMOID:
                পক্ষ GGML_UNARY_OP_HARDSIGMOID:
                পক্ষ GGML_UNARY_OP_HARDSWISH:
                পক্ষ GGML_UNARY_OP_GELU_QUICK:
                পক্ষ GGML_UNARY_OP_TANH:
                পক্ষ GGML_UNARY_OP_EXP:
                পক্ষ GGML_UNARY_OP_ELU:
                পক্ষ GGML_UNARY_OP_SGN:
                পক্ষ GGML_UNARY_OP_STEP:
                পক্ষ GGML_UNARY_OP_GELU_ERF:
                পক্ষ GGML_UNARY_OP_SOFTPLUS:
                    প্রত্যাগম সত্য;
                স্বভাব:
                    প্রত্যাগম অসত্য;
            }
        পক্ষ GGML_OP_GLU:
            বিকল্প (ggml_get_glu_op(op)) {
                পক্ষ GGML_GLU_OP_REGLU:
                পক্ষ GGML_GLU_OP_GEGLU:
                পক্ষ GGML_GLU_OP_SWIGLU:
                পক্ষ GGML_GLU_OP_GEGLU_ERF:
                পক্ষ GGML_GLU_OP_GEGLU_QUICK:
                পক্ষ GGML_GLU_OP_SWIGLU_CLAMP:
                    প্রত্যাগম সত্য;
                স্বভাব:
                    প্রত্যাগম অসত্য;
            }
            ভঙ্গ;
        পক্ষ GGML_OP_MUL_MAT:
            {
                বিকল্প (op->src[0]->type) {
#ifndef ASCEND_310P
                    পক্ষ GGML_TYPE_BF16:
#endif
                    পক্ষ GGML_TYPE_F16:
                    পক্ষ GGML_TYPE_F32:
                        প্রত্যাগম সত্য;
                    পক্ষ GGML_TYPE_Q8_0:
                    পক্ষ GGML_TYPE_Q4_0:
#ifdef ASCEND_310P
                        // Q4 && Q8 per group is not support on 310p device
                        প্রত্যাগম অসত্য;
#অন্যথা
                        // only support contiguous for quantized types.
                        প্রত্যাগম ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#endif
                    স্বভাব:
                        প্রত্যাগম অসত্য;
                }
            }
        পক্ষ GGML_OP_MUL_MAT_ID:
            বিকল্প (op->src[0]->type) {
                পক্ষ GGML_TYPE_F16:
                পক্ষ GGML_TYPE_F32:
                    প্রত্যাগম সত্য;
                পক্ষ GGML_TYPE_Q8_0:
                পক্ষ GGML_TYPE_Q4_0:
#ifdef ASCEND_310P
                    // Q4 && Q8 per group is not support on 310p device
                    প্রত্যাগম অসত্য;
#অন্যথা
                    // only support contiguous for quantized types.
                    প্রত্যাগম ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#endif
                স্বভাব:
                    প্রত্যাগম অসত্য;
            }
        // embedding
        পক্ষ GGML_OP_GET_ROWS:
            {
                বিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F32:
                    পক্ষ GGML_TYPE_F16:
#ifndef ASCEND_310P
                    পক্ষ GGML_TYPE_BF16:
#endif
                    পক্ষ GGML_TYPE_Q8_0:
                        প্রত্যাগম সত্য;
                    স্বভাব:
                        প্রত্যাগম অসত্য;
                }
            }
            ভঙ্গ;
        পক্ষ GGML_OP_SET_ROWS:
            {
                বিকল্প (op->type) {
                    পক্ষ GGML_TYPE_F32:
                    পক্ষ GGML_TYPE_F16:
#ifndef ASCEND_310P
                    পক্ষ GGML_TYPE_BF16:
#endif
                        প্রত্যাগম সত্য;
                    স্বভাব:
                        প্রত্যাগম অসত্য;
                }
            }
            ভঙ্গ;
        পক্ষ GGML_OP_CPY:
            {
                ggml_tensor * src = op->src[0];
#ifdef ASCEND_310P
                যদি ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                    // only support F32 and F16 on 310P.
                    প্রত্যাগম অসত্য;
                }
#অন্যথা
                যদি ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
                    // only support F32, F16 and BF16.
                    প্রত্যাগম অসত্য;
                }
#endif
                প্রত্যাগম সত্য;
            }
            ভঙ্গ;
        পক্ষ GGML_OP_CONT:
            {
                বিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F32:
                    পক্ষ GGML_TYPE_F16:
#ifndef ASCEND_310P
                    পক্ষ GGML_TYPE_BF16:
#endif
                        প্রত্যাগম সত্য;
                    স্বভাব:
                        প্রত্যাগম অসত্য;
                }
            }
        পক্ষ GGML_OP_ROPE:
            {
                যদি (((ধ্রুব int32_t *) op->op_params)[15] != 0) {
                    প্রত্যাগম অসত্য; // FIXME: support ggml_rope_set_offset
                }
                যদি (op->src[0]->ne[0] > 896) {
                    প্রত্যাগম অসত্য;
                }
#ifdef ASCEND_310P
                // TODO: Support rope_dim < ne00(dim)
                যদি (op->src[0]->ne[0] != op->op_params[1]) {
                    প্রত্যাগম অসত্য;
                }
                যদি (!ggml_is_contiguous(op->src[0])) {
                    প্রত্যাগম অসত্য;
                }
#endif
                প্রত্যাগম সত্য;
            }
        পক্ষ GGML_OP_UPSCALE:
            {
                // aclnnUpsampleNearest2dGetWorkspaceSize not support
                // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
                যদি (op->src[0]->ne[2] * op->ne[3] != op->src[0]->ne[3] * op->ne[2]) {
                    প্রত্যাগম অসত্য;
                }
                যদি (op->op_params[0] != GGML_SCALE_MODE_NEAREST) {
                    প্রত্যাগম অসত্য;
                }
                যদি (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                    প্রত্যাগম অসত্য;
                }
                প্রত্যাগম সত্য;
            }
        পক্ষ GGML_OP_POOL_2D:
            {
                ধ্রুব int32_t * opts = (ধ্রুব int32_t *) op->op_params;
#ifdef ASCEND_310P
                গণনা ggml_op_pool opt = static_cast<ggml_op_pool>(opts[0]);
                যদি (opt == GGML_OP_POOL_MAX) {
                    প্রত্যাগম অসত্য;
                }
#endif
                ধ্রুব পূর্ণাঙ্ক k0 = opts[1];
                ধ্রুব পূর্ণাঙ্ক k1 = opts[2];
                ধ্রুব পূর্ণাঙ্ক p0 = opts[5];
                ধ্রুব পূর্ণাঙ্ক p1 = opts[6];
                // value of paddingH should be at most half of kernelH
                // value of paddingW should be at most half of kernelW
                প্রত্যাগম (p0 <= (k0 / 2)) && (p1 <= (k1 / 2));
            }
        পক্ষ GGML_OP_SUM:
            প্রত্যাগম ggml_is_contiguous_rows(op->src[0]);
        পক্ষ GGML_OP_L2_NORM:
        পক্ষ GGML_OP_CROSS_ENTROPY_LOSS:
        পক্ষ GGML_OP_DUP:
        পক্ষ GGML_OP_IM2COL:
        পক্ষ GGML_OP_CONCAT:
        পক্ষ GGML_OP_REPEAT:
        পক্ষ GGML_OP_NONE:
        পক্ষ GGML_OP_RESHAPE:
        পক্ষ GGML_OP_VIEW:
        পক্ষ GGML_OP_PERMUTE:
        পক্ষ GGML_OP_TRANSPOSE:
        পক্ষ GGML_OP_NORM:
        পক্ষ GGML_OP_ADD:
        পক্ষ GGML_OP_ADD1:
        পক্ষ GGML_OP_SUB:
        পক্ষ GGML_OP_MUL:
        পক্ষ GGML_OP_DIV:
        পক্ষ GGML_OP_RMS_NORM:
        পক্ষ GGML_OP_SQR:
        পক্ষ GGML_OP_SQRT:
        পক্ষ GGML_OP_CLAMP:
        পক্ষ GGML_OP_DIAG_MASK_INF:
        পক্ষ GGML_OP_SUM_ROWS:
        পক্ষ GGML_OP_ARGSORT:
        পক্ষ GGML_OP_ACC:
        পক্ষ GGML_OP_SET:
        পক্ষ GGML_OP_GROUP_NORM:
            প্রত্যাগম সত্য;
        পক্ষ GGML_OP_PAD:
            // TODO: add circular padding support for cann, see https://github.com/ggml-org/llama.cpp/pull/16985
            প্রত্যাগম ggml_get_op_params_i32(op, 8) == 0;
        পক্ষ GGML_OP_ARANGE:
        পক্ষ GGML_OP_TIMESTEP_EMBEDDING:
        পক্ষ GGML_OP_LEAKY_RELU:
        পক্ষ GGML_OP_ARGMAX:
        পক্ষ GGML_OP_COS:
        পক্ষ GGML_OP_SIN:
        পক্ষ GGML_OP_LOG:
        পক্ষ GGML_OP_MEAN:
        পক্ষ GGML_OP_PAD_REFLECT_1D:
        পক্ষ GGML_OP_গুনতি_EQUAL:
        পক্ষ GGML_OP_GATED_LINEAR_ATTN:
            প্রত্যাগম সত্য;
        পক্ষ GGML_OP_OUT_PROD:
            {
#ifdef ASCEND_310P
                // Ger is not supported on 310p device
                প্রত্যাগম অসত্য;
#অন্যথা
                বিকল্প (op->src[0]->type) {
                    পক্ষ GGML_TYPE_F16:
                    পক্ষ GGML_TYPE_F32:
                        প্রত্যাগম সত্য;
                    স্বভাব:
                        প্রত্যাগম অসত্য;
                }
#endif
            }
        পক্ষ GGML_OP_CONV_TRANSPOSE_1D:
            প্রত্যাগম সত্য;
        পক্ষ GGML_OP_SCALE:
            ভিন্ন bias;
            memcpy(&bias, (ধ্রুব ভিন্ন *) (op->op_params) + 1, পরিমাণ(ভিন্ন));
            প্রত্যাগম bias == 0.0f;  // TODO: support bias != 0.0f
        পক্ষ GGML_OP_SOFT_MAX:
            // TODO: support attention sinks [TAG_ATTN_SINKS]
            যদি (op->src[2]) {
                প্রত্যাগম অসত্য;
            }
            প্রত্যাগম সত্য;
        পক্ষ GGML_OP_FLASH_ATTN_EXT:
            {
#ifdef ASCEND_310P
                // FA not support on 310p device
                প্রত্যাগম অসত্য;
#অন্যথা
                // derived from [ggml-cuda.cu]
                যদি (op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16) {
                    প্রত্যাগম অসত্য;
                }
                যদি (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32 &&
                    op->src[1]->type != GGML_TYPE_BF16) {
                    প্রত্যাগম অসত্য;
                }
                যদি (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16) {
                    প্রত্যাগম অসত্য;
                }
                // TODO: support attention sinks [TAG_ATTN_SINKS]
                যদি (op->src[4]) {
                    প্রত্যাগম অসত্য;
                }
                যদি (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                    // different head sizes of K and V are not supported yet
                    প্রত্যাগম অসত্য;
                }
                ভিন্ন logitSoftcap = 0.0f;
                memcpy(&logitSoftcap, (ধ্রুব ভিন্ন *) (op->op_params) + 2, পরিমাণ(ভিন্ন));
                যদি (logitSoftcap != 0.0f) {
                    প্রত্যাগম অসত্য;
                }
                প্রত্যাগম সত্য;
#endif
            }
        পক্ষ GGML_OP_SSM_CONV:
            প্রত্যাগম সত্য;
        পক্ষ GGML_OP_CUMSUM:
            প্রত্যাগম op->src[0]->type == GGML_TYPE_F32;
        পক্ষ GGML_OP_TRI:
            প্রত্যাগম op->src[0]->type == GGML_TYPE_F32;
        পক্ষ GGML_OP_FILL:
            প্রত্যাগম op->src[0]->type == GGML_TYPE_F32;
        পক্ষ GGML_OP_DIAG:
            প্রত্যাগম op->src[0]->type == GGML_TYPE_F32;
        পক্ষ GGML_OP_SOLVE_TRI:
            প্রত্যাগম op->src[0]->type == GGML_TYPE_F32;
        স্বভাব:
            প্রত্যাগম অসত্য;
    }

    GGML_UNUSED(dev);
}

/**
 * @brief Records an event on the CANN backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
স্থির শূন্য ggml_backend_cann_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtRecordEvent((aclrtEvent) event->context, cann_ctx->stream()));
}

/**
 * @brief Waits for a recorded event to complete on the CANN backend stream.
 *
 * This function makes the given backend wait for the event to complete on its
 * ACL runtime stream.
 *
 * @param backend Pointer to the backend structure.
 * @param event Pointer to the event structure that the backend needs to wait
 * for.
 */
স্থির শূন্য ggml_backend_cann_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    যদি (ggml_backend_is_cann(backend)) {
        ACL_CHECK(aclrtStreamWaitEvent(cann_ctx->stream(), (aclrtEvent) event->context));
    } অন্যথা {
        GGML_ABORT("fatal error");
    }
}

/**
 * @brief Structure defining the interface for the CANN backend.
 *
 * This structure contains function pointers for various operations
 * supported by the CANN backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
স্থির ধ্রুব ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .set_tensor_2d_async     = */ রিক্ত,
    /* .get_tensor_2d_async     = */ রিক্ত,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ রিক্ত,
    /* .graph_plan_free         = */ রিক্ত,
    /* .graph_plan_update       = */ রিক্ত,
    /* .graph_plan_compute      = */ রিক্ত,
    /* .graph_compute           = */ ggml_backend_cann_graph_compute,
    /* .event_record            = */ ggml_backend_cann_event_record,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .graph_optimize          = */ রিক্ত,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
স্থির ggml_guid_t ggml_backend_cann_guid() {
    স্থির ggml_guid guid = { 0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                              0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64 };
    প্রত্যাগম &guid;
}

// backend device
সংরচনা ggml_backend_cann_device_context {
    পূর্ণাঙ্ক         device;
    std::string name;
    std::string description;
    পূর্ণাঙ্ক op_offload_min_batch_size;
};

স্থির ধ্রুব বর্ণ * ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    প্রত্যাগম ctx->name.c_str();
}

স্থির ধ্রুব বর্ণ * ggml_backend_cann_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    প্রত্যাগম ctx->description.c_str();
}

স্থির শূন্য ggml_backend_cann_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ggml_backend_cann_get_device_memory(ctx->device, মুক্ত, total);
}

স্থির গণনা ggml_backend_dev_type ggml_backend_cann_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    প্রত্যাগম GGML_BACKEND_DEVICE_TYPE_GPU;
}

স্থির শূন্য ggml_backend_cann_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type        = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_memory(dev, &props->memory_free, &props->memory_total);

    তার্কিক host_buffer = getenv("GGML_CANN_NO_PINNED") == nullptr;

    props->caps = {
        /* .async                 = */ অসত্য,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ অসত্য,
        /* .events                = */ সত্য,
        /* .mmap_support          = */ সত্য,
    };
}

স্থির ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev, ধ্রুব বর্ণ * params) {
    GGML_UNUSED(params);
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    প্রত্যাগম ggml_backend_cann_init(ctx->device);
}

/**
 * @brief Checks if the CANN backend supports a specific backend buffer type.
 *
 * This function determines whether the CANN backend supports the given backend
 * buffer type by comparing the device context of the backend and buffer type.
 * It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the CANN backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the CANN backend supports the buffer type,
 *              otherwise false.
 */
স্থির তার্কিক ggml_backend_cann_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    যদি (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context *      dev_ctx  = (ggml_backend_cann_device_context *) dev->context;
        ggml_backend_cann_buffer_type_context * buft_ctx = (ggml_backend_cann_buffer_type_context *) buft->context;
        প্রত্যাগম buft_ctx->device == dev_ctx->device;
    }
    প্রত্যাগম অসত্য;
}

স্থির ggml_backend_buffer_type_t ggml_backend_cann_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    প্রত্যাগম ggml_backend_cann_buffer_type(ctx->device);
}

স্থির ggml_backend_buffer_type_t ggml_backend_cann_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    প্রত্যাগম ggml_backend_cann_host_buffer_type();
}

/**
 * @brief Determines if a tensor operation should be offloaded to the CANN
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * CANN backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the CANN backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
স্থির তার্কিক ggml_backend_cann_offload_op(ggml_backend_dev_t dev, ধ্রুব ggml_tensor * op) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;

    প্রত্যাগম op->ne[1] >= dev_ctx->op_offload_min_batch_size && op->op != GGML_OP_GET_ROWS;
}

/**
 * @brief Creates a new event for the CANN backend device.
 *
 * This function initializes a new event for the CANN backend by setting the
 * device and creating an ACL runtime event. The created event is then wrapped
 * in a ggml_backend_event structure and returned.
 *
 * @param backend Pointer to the CANN backend.
 * @return ggml_backend_event_t Returns a pointer to the new event structure.
 */
স্থির ggml_backend_event_t ggml_backend_cann_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *) dev->context;

    ggml_cann_set_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    প্রত্যাগম নূতন ggml_backend_event{
        /* .device = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), dev_ctx->device),
        /* .context = */ event,
    };
}

/**
 * @brief Frees a CANN backend event.
 *
 * This function destroys the ACL runtime event associated with the given CANN
 * backend event and then deletes the event structure itself.
 *
 * @param event Pointer to the event structure to be freed.
 */
স্থির শূন্য ggml_backend_cann_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent) event->context));

    delete event;
    GGML_UNUSED(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
স্থির শূন্য ggml_backend_cann_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent) event->context));

    GGML_UNUSED(dev);
}

স্থির ধ্রুব ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_memory,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,  // called for every card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cann_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ রিক্ত,  // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};

// backend reg
সংরচনা ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

স্থির ধ্রুব বর্ণ * ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    প্রত্যাগম GGML_CANN_NAME;
}

স্থির size_t ggml_backend_cann_reg_get_device_গুনতি(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    প্রত্যাগম ctx->devices.size();
}

স্থির ggml_backend_dev_t ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    প্রত্যাগম ctx->devices[index];
}

স্থির শূন্য * ggml_backend_cann_reg_get_proc_address(ggml_backend_reg_t reg, ধ্রুব বর্ণ * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    // reserved for future use
    প্রত্যাগম nullptr;
}

স্থির ধ্রুব ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_গুনতি,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_address,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    স্থির ggml_backend_reg reg;
    স্থির তার্কিক             initialized = অসত্য;

    {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        যদি (!initialized) {
            aclInit(nullptr);
            ggml_backend_cann_reg_context * ctx = নূতন ggml_backend_cann_reg_context;
            ধ্রুব পূর্ণাঙ্ক min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            প্রতি (পূর্ণাঙ্ক i = 0; i < ggml_cann_info().device_গুনতি; i++) {
                ggml_backend_cann_device_context * dev_ctx = নূতন ggml_backend_cann_device_context();
                dev_ctx->description                       = aclrtGetSocName();
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_CANN_NAME + std::to_string(i);
                dev_ctx->op_offload_min_batch_size         = min_batch_size;
                ggml_cann_set_device(i);
                ggml_backend_dev_t dev = নূতন ggml_backend_device{ /* .iface   = */ ggml_backend_cann_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_cann_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = সত্য;
    }

    প্রত্যাগম &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(nullptr);
    যদি (device < 0 || device >= ggml_backend_cann_get_device_গুনতি()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        প্রত্যাগম nullptr;
    }

    ggml_backend_cann_context * ctx = নূতন ggml_backend_cann_context(device);
    যদি (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        প্রত্যাগম nullptr;
    }
    ggml_cann_set_device(ctx->device);
    ggml_backend_t cann_backend =
        নূতন ggml_backend{ /* .guid      = */ ggml_backend_cann_guid(),
                          /* .interface = */ ggml_backend_cann_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                          /* .context   = */ ctx };

    প্রত্যাগম cann_backend;
}

তার্কিক ggml_backend_is_cann(ggml_backend_t backend) {
    প্রত্যাগম backend != রিক্ত && ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_গুনতি() {
    প্রত্যাগম ggml_cann_info().device_গুনতি;
}

শূন্য ggml_backend_cann_get_device_description(int32_t device, বর্ণ * description, size_t description_size) {
    ggml_cann_set_device(device);
    ধ্রুব বর্ণ * soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name);
}

শূন্য ggml_backend_cann_get_device_memory(int32_t device, size_t * মুক্ত, size_t * total) {
    ggml_cann_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, মুক্ত, total));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)

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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cann.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cann/aclnn_ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cann/common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <acl/acl.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <aclnnop/aclnn_trans_matmul_weight.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdarg.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <chrono>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>
#ਅਨ੍ਤਰ੍ਭਾਵ <queue>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] ਖਾਲੀ ggml_cann_error(ਅਚਲ ਅੱਖਰ * stmt, ਅਚਲ ਅੱਖਰ * func, ਅਚਲ ਅੱਖਰ * file, ਪੂਰਨ ਸਤਰ, ਅਚਲ ਅੱਖਰ * ਸੁਨੇਹਾ_2) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", ਸੁਨੇਹਾ_2);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, ਸਤਰ);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

// Thread-local variable to record the current device of this thread.
thread_local ਪੂਰਨ g_current_cann_device = -1;

/**
 * @brief Set the CANN device to be used.
 *
 * @param device The target device ID to set.
 */
ਖਾਲੀ ggml_cann_ਸੈੱਟ_device(ਅਚਲ int32_t device) {
    // int current_device = -1;
    // Note: In some CANN versions, if no device has been set yet,
    //       aclrtGetDevice(&current_device) may return 0 by default.
    // aclrtGetDevice(&current_device);

    // If the current device is already the target one, no need to switch.
    ਜੇ (device == g_current_cann_device) {
        ਵਾਪਸ;
    }

    // Switch to the new device.
    ACL_CHECK(aclrt_ਸੈੱਟ_Device(device));

    // Update the global device record.
    g_current_cann_device = device;
}

/**
 * @brief Get the value of the specified environment variable (name) as lowercase.
 *        if not empty, return a std::string object
 */
std::optional<std::string> get_env_as_lowercase(ਅਚਲ std::string & name) {
    ਅਚਲ ਅੱਖਰ * val = std::getenv(name.c_str());
    ਜੇ (!val) {
        ਵਾਪਸ std::nullopt;
    }
    std::string res = std::string(val);
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    ਵਾਪਸ res;
}

/**
 * @brief Verify whether the environment variable is a valid value.
 */
ਤਾਰਕਿਕ parse_bool(ਅਚਲ std::string & value) {
    ਸਥਿਰ ਅਚਲ std::unordered_set<std::string> valid_ਮੁੱਲਾਂ = { "on", "1", "yes", "y", "enable", "true" };
    ਵਾਪਸ valid_ਮੁੱਲਾਂ.find(value) != valid_ਮੁੱਲਾਂ.end();
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
ਪੂਰਨ parse_integer(ਅਚਲ std::string & value) {
    ਕੋਸ਼ਿਸ਼ {
        ਵਾਪਸ std::stoi(value);
    } ਫੜੋ (...) {
        ਵਾਪਸ 0;
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
ਸਥਿਰ ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrt_Get_Device_ਗਿਣਤੀ((uint32_t *) &info.device_ਗਿਣਤੀ);

    ਜੇ (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n", __func__, acl_Get_Recent_Err_ਸੁਨੇਹਾ());
        ਵਾਪਸ info;
    }

    GGML_ASSERT(info.device_ਗਿਣਤੀ <= GGML_CANN_MAX_DEVICES);

    ਲਈ (ਪੂਰਨ id = 0; id < info.device_ਗਿਣਤੀ; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
        prop.ਵੰਡ_Type       = ACL_MEM_ਵੰਡ_TYPE_PINNED;
        prop.memAttr              = ACL_HBM_MEM_HUGE;
        prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = id;
        prop.reserve              = 0;
        err                       = aclrt_Mem_Get_ਵੰਡ_Granularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                                                                     &info.devices[id].vmm_granularity);
        info.devices[id].vmm      = err == ACL_SUCCESS;

        size_t ਮੁਕਤ, ਕੁੱਲ;
        ggml_backend_cann_get_device_ਸਮ੍ਰਿਤੀ(id, &ਮੁਕਤ, &ਕੁੱਲ);
        info.devices[id].ਕੁੱਲ_vram = ਮੁਕਤ;
    }

    // TODO: add more device info later.
    ਵਾਪਸ info;
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
ਅਚਲ ggml_cann_device_info & ggml_cann_info() {
    ਸਥਿਰ ggml_cann_device_info info = ggml_cann_init();
    ਵਾਪਸ info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(priority segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
ਬਣਤਰ ggml_cann_pool_buf_prio : ਜਨਤਕ ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    ਸਥਿਰ ਅਚਲ size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    ਸਥਿਰ ਅਚਲ size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    ਸਥਿਰ ਅਚਲ size_t alignment = 128;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    ਪੂਰਨ device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    ਤਾਰਕਿਕ disable_clean = ਝੂਠ;

    /**
     * @brief Structure representing a CANN buffer.
     */
    ਬਣਤਰ ggml_cann_ਬਫਰ {
        ਖਾਲੀ *                                ptr  = ਖਾਲੀ_ਸੂਚਕ;  ///< Pointer to the buffer.
        size_t                                size = 0;        ///< Size of the buffer.
        std::chrono::steady_clock::time_point ਆਖਰੀ_ਵਰਤਿਆ;       ///< Last used time.

        ਤਾਰਕਿਕ ਸੰਚਾਲਕ>(ਅਚਲ ggml_cann_ਬਫਰ & other) ਅਚਲ { ਵਾਪਸ size > other.size; }
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    std::unordered_map<ਖਾਲੀ *, size_t>                                                   ਬਫਰ_pool;
    std::priority_queue<ggml_cann_ਬਫਰ, std::vector<ggml_cann_ਬਫਰ>, std::greater<>> free_buffers;

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_ਮਾਪ = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    ਸਪਸ਼ਟ ggml_cann_pool_buf_prio(ਪੂਰਨ device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf_prio() {
        ggml_cann_ਸੈੱਟ_device(device);
        ਲਈ (ਸਵੈਚਾਲਿਤ & [b_ptr, b_ਮਾਪ] : ਬਫਰ_pool) {
            aclrtFree(b_ptr);
            pool_ਮਾਪ -= b_ਮਾਪ;
        }
        ਬਫਰ_pool.clear();
        GGML_ASSERT(pool_ਮਾਪ == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    ਖਾਲੀ * alloc(size_t size, size_t * actual_ਮਾਪ) override {
        size = GGML_PAD(size, alignment);
        ਜੇ (size == 0) {
            size = alignment;
        }

        ਖਾਲੀ * ptr = ਖਾਲੀ_ਸੂਚਕ;
        ਸਵੈਚਾਲਿਤ   now = std::chrono::steady_clock::now();

        std::vector<ggml_cann_ਬਫਰ> free_buffers_rest;
        free_buffers_rest.reserve(free_buffers.size());
        ਜਦੋਂ (!free_buffers.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਸਵੈਚਾਲਿਤ b = free_buffers.top();
            free_buffers.ਕੱਢੋ();

            ਜੇ (b.size >= size) {
                // reuse the buffer if the size is enough
                ਅਚਲ size_t margin = b.size - size;
                ਜੇ (margin <= max_reuse_margin) {
                    *actual_ਮਾਪ = b.size;
                    ptr          = b.ptr;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
                    ਰੋਕੋ;
                }
            }

            ਤਾਰਕਿਕ should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.ਆਖਰੀ_ਵਰਤਿਆ).count() > 100;
            ਜੇ (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_ਮਾਪ -= b.size;
                ਬਫਰ_pool.erase(b.ptr);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
                ਜਾਰੀ;
            }
            free_buffers_rest.push_back(b);
        }
        ਲਈ (ggml_cann_ਬਫਰ & b : free_buffers_rest) {
            free_buffers.ਧੱਕੋ(std::move(b));
        }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d] free pool_size = %5u MB\n\n", device,
                      (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
        ਜੇ (ptr != ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ptr;
        }

        // allocate a new buffer if no buffer can be reused
        ggml_cann_ਸੈੱਟ_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));
        *actual_ਮਾਪ = size;
        pool_ਮਾਪ += size;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: allocate %p, "
            "pool_size = %5u MB, "
            "size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576),
            (uint32_t) (GGML_PAD(size, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
        ਬਫਰ_pool.emplace(ptr, size);
        ਵਾਪਸ ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size) override {
        GGML_ਅਣਵਰਤਿਆ(size);
        ਸਵੈਚਾਲਿਤ it = ਬਫਰ_pool.find(ptr);
        ਜੇ (it == ਬਫਰ_pool.end()) {
            GGML_ABORT("cann pool[%d]: buffer %p not found in pool\n", device, ptr);
        }

        ਸਵੈਚਾਲਿਤ now = std::chrono::steady_clock::now();
        free_buffers.emplace(ggml_cann_ਬਫਰ{ ptr, it->second, now });
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: return   %p, "
            "pool_size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
    }
};

/**
 * @brief A pool of CANN buffers(segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
ਬਣਤਰ ggml_cann_pool_buf : ਜਨਤਕ ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    ਸਥਿਰ ਅਚਲ size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    ਸਥਿਰ ਅਚਲ size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    ਸਥਿਰ ਅਚਲ size_t alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    ਸਥਿਰ ਅਚਲ ਪੂਰਨ MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    ਪੂਰਨ device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    ਤਾਰਕਿਕ disable_clean = ਝੂਠ;

    /**
     * @brief Structure representing a CANN buffer.
     */
    ਬਣਤਰ ggml_cann_ਬਫਰ {
        ਖਾਲੀ *                                ptr  = ਖਾਲੀ_ਸੂਚਕ;  ///< Pointer to the buffer memory.
        size_t                                size = 0;        ///< Size of the buffer.
        ਤਾਰਕਿਕ                                  ਵਰਤਿਆ = ਝੂਠ;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point ਆਖਰੀ_ਵਰਤਿਆ;       ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_ਬਫਰ ਬਫਰ_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_ਮਾਪ = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    ਸਪਸ਼ਟ ggml_cann_pool_buf(ਪੂਰਨ device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf() {
        ggml_cann_ਸੈੱਟ_device(device);
        ਲਈ (ਪੂਰਨ i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_ਬਫਰ & b = ਬਫਰ_pool[i];
            ਜੇ (b.ptr != ਖਾਲੀ_ਸੂਚਕ) {
                aclrtFree(b.ptr);
                pool_ਮਾਪ -= b.size;
            }
        }
        GGML_ASSERT(pool_ਮਾਪ == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    ਖਾਲੀ * alloc(size_t size, size_t * actual_ਮਾਪ) override {
        size = GGML_PAD(size, alignment);
        ਜੇ (size == 0) {
            size = alignment;
        }

        ਖਾਲੀ * ptr = ਖਾਲੀ_ਸੂਚਕ;
        ਸਵੈਚਾਲਿਤ   now = std::chrono::steady_clock::now();

        ਪੂਰਨ i = 0;
        ਲਈ (; i < MAX_BUFFERS; ++i) {
            ggml_cann_ਬਫਰ & b = ਬਫਰ_pool[i];
            ਜੇ (b.ptr == ਖਾਲੀ_ਸੂਚਕ) {
                ਰੋਕੋ;
            }
            ਜੇ (b.ਵਰਤਿਆ) {
                ਜਾਰੀ;
            }
            ਜੇ (b.size >= size) {
                // reuse the buffer if the size is enough
                ਅਚਲ size_t margin = b.size - size;
                ਜੇ (margin <= max_reuse_margin) {
                    *actual_ਮਾਪ = b.size;
                    b.ਵਰਤਿਆ       = ਸੱਚ;
                    ptr          = b.ptr;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
                    ਰੋਕੋ;
                }
            }

            ਤਾਰਕਿਕ should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.ਆਖਰੀ_ਵਰਤਿਆ).count() > 100;
            ਜੇ (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_ਮਾਪ -= b.size;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
                b.ptr = ਖਾਲੀ_ਸੂਚਕ;
            }
        }
        ਜੇ (ptr != ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ptr;
        }

        ਜੇ (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_cann_ਬਫਰ & b = ਬਫਰ_pool[i];
            ggml_cann_ਸੈੱਟ_device(device);
            ACL_CHECK(aclrtMalloc(&b.ptr, size, ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));
            pool_ਮਾਪ += size;
            *actual_ਮਾਪ = size;
            b.size       = size;
            b.ਵਰਤਿਆ       = ਸੱਚ;
            ਜੇ (i >= MAX_BUFFERS - 8) {
                GGML_LOG_WARN("cann pool[%d]: slots almost full\n", device);
            }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576),
                (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
            ਵਾਪਸ b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size) override {
        GGML_ਅਣਵਰਤਿਆ(size);
        ਲਈ (ਪੂਰਨ i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_ਬਫਰ & b = ਬਫਰ_pool[i];
            ਜੇ (b.ptr != ptr) {
                ਜਾਰੀ;
            }
            b.ਵਰਤਿਆ      = ਝੂਠ;
            b.ਆਖਰੀ_ਵਰਤਿਆ = std::chrono::steady_clock::now();
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_ਮਾਪ, 1048576) / 1048576));
#ਜੇ_ਅੰਤ
            ਵਾਪਸ;
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
ਬਣਤਰ ggml_cann_pool_vmm : ਜਨਤਕ ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    size_t max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    ਪੂਰਨ device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    ਖਾਲੀ * pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_ਵਰਤਿਆ = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_ਮਾਪ = 0;

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
    std::vector<ਖਾਲੀ *> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    ਸਪਸ਼ਟ ggml_cann_pool_vmm(ਪੂਰਨ device) : device(device) {
        ਸਵੈਚਾਲਿਤ dev    = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size    = dev.ਕੁੱਲ_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        ਜੇ (pool_addr != 0) {
            ਲਈ (ਸਵੈਚਾਲਿਤ & ਵਿੱਥ : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(ਵਿੱਥ));
            }
            ਲਈ (ਸਵੈਚਾਲਿਤ & handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrt_Release_Mem_ਪਤਾ(pool_addr));
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
    ਖਾਲੀ * alloc(size_t size, size_t * actual_ਮਾਪ) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        ਅਚਲ size_t alignment = 128;
        size                   = GGML_PAD(size, alignment);
        ਜੇ (size == 0) {
            size = alignment;
        }

        size_t avail = pool_ਮਾਪ - pool_ਵਰਤਿਆ;

        ਜੇ (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_ਮਾਪ = size - avail;
            reserve_ਮਾਪ        = GGML_PAD(reserve_ਮਾਪ, granularity);

            GGML_ASSERT(pool_ਮਾਪ + reserve_ਮਾਪ <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
            prop.ਵੰਡ_Type       = ACL_MEM_ਵੰਡ_TYPE_PINNED;
            prop.memAttr              = ACL_HBM_MEM_HUGE;
            prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id          = device;
            prop.reserve              = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_ਮਾਪ, &prop, 0));

            // reserve virtual address space (if not already reserved)
            ਜੇ (pool_addr == 0) {
                ACL_CHECK(aclrt_Reserve_Mem_ਪਤਾ(&pool_addr, max_size, 0, ਨਿਰਾ, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((ਅੱਖਰ *) pool_addr + pool_ਮਾਪ, reserve_ਮਾਪ, 0, handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((ਅੱਖਰ *) pool_addr + pool_ਮਾਪ);

            // add to the pool
            pool_ਮਾਪ += reserve_ਮਾਪ;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
            GGML_LOG_INFO("cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n", device,
                          (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) (pool_ਮਾਪ / 1024 / 1024),
                          (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) (reserve_ਮਾਪ / 1024 / 1024));
#ਜੇ_ਅੰਤ
        }

        GGML_ASSERT(pool_addr != 0);

        ਖਾਲੀ * ptr   = (ਖਾਲੀ *) ((ਅੱਖਰ *) pool_addr + pool_ਵਰਤਿਆ);
        *actual_ਮਾਪ = size;
        pool_ਵਰਤਿਆ += size;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) size,
                      (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ptr);
#ਜੇ_ਅੰਤ
        ਵਾਪਸ ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size) override {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) size,
                      (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) ptr);
#ਜੇ_ਅੰਤ

        pool_ਵਰਤਿਆ -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (ਖਾਲੀ *) ((ਅੱਖਰ *) pool_addr + pool_ਵਰਤਿਆ));
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
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(ਪੂਰਨ device) {
    std::string mem_pool_type = get_env_as_lowercase("GGML_CANN_MEM_POOL").value_or("");

    ਜੇ (mem_pool_type == "prio") {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        ਵਾਪਸ std::unique_ptr<ggml_cann_pool>(ਨਵਾਂ ggml_cann_pool_buf_prio(device));
    }

    ਜੇ (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        ਵਾਪਸ std::unique_ptr<ggml_cann_pool>(ਨਵਾਂ ggml_cann_pool_vmm(device));
    }

    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    ਵਾਪਸ std::unique_ptr<ggml_cann_pool>(ਨਵਾਂ ggml_cann_pool_buf(device));
}

// cann buffer

/**
 * @brief Tracks multi-threaded write progress for a single tensor.
 *
 * When multiple threads call set_tensor on different chunks of the same tensor,
 * this tracker accumulates progress and defers post-processing (quantized format
 * transform or ND-to-NZ conversion) until all data has been written.
 */
ਬਣਤਰ Tensor_ਸੈੱਟ_Tracker {
    std::mutex mtx;                   ///< Protects concurrent access to this tracker
    size_t ਬਾਈਟਾਂ_written = 0;         ///< Accumulated bytes written so far
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ = 0;           ///< Target size (full tensor)
    std::vector<uint8_t> host_ਬਫਰ; ///< Host staging buffer for quantized tensors
};

/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
ਬਣਤਰ ggml_backend_cann_ਬਫਰ_context {
    int32_t device;             ///< The device ID associated with this buffer context.
    ਖਾਲੀ *  dev_ptr = ਖਾਲੀ_ਸੂਚਕ;  ///< Pointer to the device memory allocated for the buffer.

    std::mutex tracker_mutex;   ///< Protects the trackers map
    std::unordered_map<ਖਾਲੀ *, std::unique_ptr<Tensor_ਸੈੱਟ_Tracker>> trackers;

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_ਬਫਰ_context(int32_t device, ਖਾਲੀ * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_ਬਫਰ_context() { ACL_CHECK(aclrtFree(dev_ptr)); }

    /**
     * @brief Get or create a tracker for the given tensor.
     */
    Tensor_ਸੈੱਟ_Tracker * get_or_create_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> ਤਾਲਾ(tracker_mutex);
        ਸਵੈਚਾਲਿਤ ਕੁੰਜੀ = tensor->ਅੰਕੜੇ;
        ਸਵੈਚਾਲਿਤ it = trackers.find(ਕੁੰਜੀ);
        ਜੇ (it == trackers.end()) {
            ਸਵੈਚਾਲਿਤ tracker = std::make_unique<Tensor_ਸੈੱਟ_Tracker>();
            tracker->ਕੁੱਲ_ਬਾਈਟਾਂ = ggml_nbytes(tensor);
            ਸਵੈਚਾਲਿਤ * ptr = tracker.get();
            trackers[ਕੁੰਜੀ] = std::move(tracker);
            ਵਾਪਸ ptr;
        }
        ਵਾਪਸ it->second.get();
    }

    /**
     * @brief Remove the tracker for the given tensor.
     */
    ਖਾਲੀ remove_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> ਤਾਲਾ(tracker_mutex);
        trackers.erase(tensor->ਅੰਕੜੇ);
    }
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
ਬਣਤਰ ggml_backend_cann_ਬਫਰ_type_context {
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
ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cann_ਬਫਰ_type_name(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_cann_ਬਫਰ_type_context * buft_ctx = (ggml_backend_cann_ਬਫਰ_type_context *) buft->context;

    ਵਾਪਸ buft_ctx->name.c_str();
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
ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_buft_is_cann(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ buft->iface.get_name == ggml_backend_cann_ਬਫਰ_type_name;
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_cann_ਬਫਰ_context * ctx = (ggml_backend_cann_ਬਫਰ_context *) ਬਫਰ->context;
    ਮਿਟਾਓ ctx;
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
ਸਥਿਰ ਖਾਲੀ * ggml_backend_cann_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_cann_ਬਫਰ_context * ctx = (ggml_backend_cann_ਬਫਰ_context *) ਬਫਰ->context;
    ਵਾਪਸ ctx->dev_ptr;
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_transform_q4_0(ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * src, ਖਾਲੀ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_ਬਾਈਟਾਂ = n_elems * ਆਕਾਰ(uint8_t) / 2;

    uint8_t *  quant_ਵਿੱਥ = (uint8_t *) dst;
    uint16_t * scale_ਵਿੱਥ = (uint16_t *) ((ਅੱਖਰ *) dst + quant_ਬਾਈਟਾਂ);

    ਲਈ (ਪੂਰਨ i = 0; i < groups; i++) {
        ਅਚਲ ਖੰਡ_q4_0 * ਸਮੂਹ = (ਅਚਲ ਖੰਡ_q4_0 *) ((ਅਚਲ ਅੱਖਰ *) src + i * ਆਕਾਰ(ਖੰਡ_q4_0));
        *scale_ਵਿੱਥ            = ਸਮੂਹ->d;
        scale_ਵਿੱਥ++;

        // 0-15
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_ਵਿੱਥ) = (ਸਮੂਹ->qs[j] & 0x0F);
            (*quant_ਵਿੱਥ) |= ((ਸਮੂਹ->qs[j + 1] << 4));
            quant_ਵਿੱਥ++;
        }

        // 16-31
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_ਵਿੱਥ) = (ਸਮੂਹ->qs[j] >> 4);
            (*quant_ਵਿੱਥ) |= (ਸਮੂਹ->qs[j + 1] & 0xF0);
            quant_ਵਿੱਥ++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    ਲਈ (quant_ਵਿੱਥ = (uint8_t *) dst; quant_ਵਿੱਥ < (uint8_t *) dst + quant_ਬਾਈਟਾਂ; quant_ਵਿੱਥ++) {
        (*quant_ਵਿੱਥ) ^= 0x88;
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_transform_ਪਿੱਠ_q4_0(ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * src, ਖਾਲੀ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_ਬਾਈਟਾਂ = n_elems * ਆਕਾਰ(uint8_t) / 2;

    uint8_t *  quant_ਵਿੱਥ = (uint8_t *) src;
    uint16_t * scale_ਵਿੱਥ = (uint16_t *) ((ਅੱਖਰ *) src + quant_ਬਾਈਟਾਂ);

    ਲਈ (; quant_ਵਿੱਥ < (uint8_t *) src + quant_ਬਾਈਟਾਂ; quant_ਵਿੱਥ++) {
        (*quant_ਵਿੱਥ) ^= 0x88;
    }
    quant_ਵਿੱਥ = (uint8_t *) src;

    ਲਈ (ਪੂਰਨ i = 0; i < groups; i++) {
        ਖੰਡ_q4_0 * ਸਮੂਹ = (ਖੰਡ_q4_0 *) ((ਅੱਖਰ *) dst + i * ਆਕਾਰ(ਖੰਡ_q4_0));
        ਸਮੂਹ->d           = *scale_ਵਿੱਥ;
        scale_ਵਿੱਥ++;

        // 0-15
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 2; j += 2) {
            ਸਮੂਹ->qs[j]     = ((*quant_ਵਿੱਥ) & 0x0F);
            ਸਮੂਹ->qs[j + 1] = ((*quant_ਵਿੱਥ) >> 4);
            quant_ਵਿੱਥ++;
        }

        // 16-31
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 2; j += 2) {
            ਸਮੂਹ->qs[j] |= ((*quant_ਵਿੱਥ) << 4);
            ਸਮੂਹ->qs[j + 1] |= ((*quant_ਵਿੱਥ) & 0xF0);
            quant_ਵਿੱਥ++;
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_transform_q8_0(ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * src, ਖਾਲੀ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_ਬਾਈਟਾਂ = n_elems * ਆਕਾਰ(uint8_t);

    uint8_t *  quant_ਵਿੱਥ = (uint8_t *) dst;
    uint16_t * scale_ਵਿੱਥ = (uint16_t *) ((ਅੱਖਰ *) dst + quant_ਬਾਈਟਾਂ);

    ਲਈ (ਪੂਰਨ i = 0; i < groups; i++) {
        ਅਚਲ ਖੰਡ_q8_0 * ਸਮੂਹ = (ਅਚਲ ਖੰਡ_q8_0 *) ((ਅਚਲ ਅੱਖਰ *) src + i * ਆਕਾਰ(ਖੰਡ_q8_0));
        *scale_ਵਿੱਥ            = ਸਮੂਹ->d;
        scale_ਵਿੱਥ++;
        size_t ਸਮੂਹ_quant_ਮਾਪ = QK8_0 * ਆਕਾਰ(uint8_t);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(quant_ਵਿੱਥ, ਸਮੂਹ->qs, ਸਮੂਹ_quant_ਮਾਪ);
        quant_ਵਿੱਥ += ਸਮੂਹ_quant_ਮਾਪ;
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_transform_ਪਿੱਠ_q8_0(ਅਚਲ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * src, ਖਾਲੀ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_ਬਾਈਟਾਂ = n_elems * ਆਕਾਰ(uint8_t);

    ਅਚਲ uint8_t *  quant_ਵਿੱਥ = (ਅਚਲ uint8_t *) src;
    ਅਚਲ uint16_t * scale_ਵਿੱਥ = (ਅਚਲ uint16_t *) ((ਅਚਲ ਅੱਖਰ *) src + quant_ਬਾਈਟਾਂ);

    ਲਈ (ਪੂਰਨ i = 0; i < groups; i++) {
        ਖੰਡ_q8_0 * ਸਮੂਹ = (ਖੰਡ_q8_0 *) ((ਅੱਖਰ *) dst + i * ਆਕਾਰ(ਖੰਡ_q8_0));
        ਸਮੂਹ->d           = *scale_ਵਿੱਥ;
        scale_ਵਿੱਥ++;
        size_t ਸਮੂਹ_quant_ਮਾਪ = QK8_0 * ਆਕਾਰ(uint8_t);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਸਮੂਹ->qs, quant_ਵਿੱਥ, ਸਮੂਹ_quant_ਮਾਪ);
        quant_ਵਿੱਥ += ਸਮੂਹ_quant_ਮਾਪ;
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_transform(ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * src, ਖਾਲੀ * dst) {
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_transform_ਪਿੱਠ(ਅਚਲ ggml_tensor * tensor, ਖਾਲੀ * src, ਖਾਲੀ * dst) {
    ਚੋਣ (tensor->type) {
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_ਪਿੱਠ_q4_0(tensor, src, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_ਪਿੱਠ_q8_0(tensor, src, dst);
            ਰੋਕੋ;
        ਮੂਲ:
            ਰੋਕੋ;
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
ਸਥਿਰ ਤਾਰਕਿਕ need_transform(ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
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
ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_cann_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor) {
    ਜੇ (tensor->view_src != ਨਿਰਾ && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->ਬਫਰ->buft == ਬਫਰ->buft);
        ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
    }

    // TODO: cann backend doesn't support quantized yet. Just leave the code
    // here.
    ਜੇ (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_ਮਾਪ = ggml_nbytes(tensor);
        size_t padded_ਮਾਪ   = ggml_backend_buft_get_alloc_ਮਾਪ(ਬਫਰ->buft, tensor);

        ਜੇ (padded_ਮਾਪ > original_ਮਾਪ && tensor->view_src == ਖਾਲੀ_ਸੂਚਕ) {
            size_t memset_ਮਾਪ = padded_ਮਾਪ - original_ਮਾਪ;
            ACL_CHECK(aclrtMemset((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + original_ਮਾਪ, memset_ਮਾਪ, 0, memset_ਮਾਪ));
        }
    }
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

/**
 * @brief Workspace for caching NZ buffers per device.
 *
 * This struct manages a device buffer used in NZ computations. It supports
 * allocation, reallocation, and clearing of cached memory. The struct is
 * designed to be used with a global array, one per device.
 */
ਬਣਤਰ ggml_cann_nz_workspace {
    std::mutex mtx;    // Protects ptr/allocated from concurrent access
    ਖਾਲੀ * ptr;        // Pointer to allocated device buffer
    size_t allocated;  // Size of currently allocated buffer in bytes

    /**
     * @brief Constructor. Initializes the workspace with no allocated memory.
     */
    ggml_cann_nz_workspace() : ptr(ਖਾਲੀ_ਸੂਚਕ), allocated(0) {}

    /**
     * @brief Free cached memory and reset the workspace.
     *
     * If a buffer has been allocated, this function releases it using
     * aclrtFree and resets internal state.
     */
    ਖਾਲੀ clear() {
        ਜੇ (ptr) {
            ACL_CHECK(aclrtFree(ptr));
            ptr       = ਖਾਲੀ_ਸੂਚਕ;
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
    ਖਾਲੀ realloc(size_t new_ਮਾਪ) {
        ਜੇ (new_ਮਾਪ > allocated) {
            clear();
            ACL_CHECK(aclrtMalloc(&ptr, new_ਮਾਪ, ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ));
            allocated = new_ਮਾਪ;
        }
    }

    /**
     * @brief Get the device buffer pointer.
     *
     * @return Pointer to the allocated buffer, or nullptr if not allocated.
     */
    ਖਾਲੀ * get() ਅਚਲ { ਵਾਪਸ ptr; }
};

/**
 * @brief Global array of NZ workspaces, one per device.
 */
ਸਥਿਰ ggml_cann_nz_workspace g_nz_workspaces[GGML_CANN_MAX_DEVICES];

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
ਸਥਿਰ ਖਾਲੀ ਭਾਰ_format_ਮੰਜ਼ਿਲ_nz(ggml_tensor * tensor, ਪੂਰਨ device) {
    acl_tensor_ptr ਭਾਰ_Transposed = ggml_cann_create_tensor(tensor, tensor->ne, tensor->nb, 2, ACL_FORMAT_ND, 0);
    uint64_t       workspace_ਮਾਪ    = 0;
    aclOpExecutor * executor;

    // TransMatmulWeight
    ACL_CHECK(aclnn_Trans_Matmul_ਭਾਰ_Get_Workspace_ਮਾਪ(ਭਾਰ_Transposed.get(), &workspace_ਮਾਪ, &executor));

    std::lock_guard<std::mutex> ਤਾਲਾ(g_nz_workspaces[device].mtx);
    // Avoid frequent malloc/free of the workspace.
    g_nz_workspaces[device].realloc(workspace_ਮਾਪ);

    ਖਾਲੀ * g_nz_workspace = g_nz_workspaces[device].get();

    ACL_CHECK(aclnn_Trans_Matmul_ਭਾਰ(g_nz_workspace, workspace_ਮਾਪ, executor, ਖਾਲੀ_ਸੂਚਕ));
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                ggml_tensor *         tensor,
                                                ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                                size_t                ਵਿੱਥ,
                                                size_t                size) {
    ggml_backend_cann_ਬਫਰ_context * ctx = (ggml_backend_cann_ਬਫਰ_context *) ਬਫਰ->context;

    ggml_cann_ਸੈੱਟ_device(ctx->device);

    // Only check env once.
    ਸਥਿਰ ਤਾਰਕਿਕ ਭਾਰ_ਮੰਜ਼ਿਲ_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    ਤਾਰਕਿਕ is_quantized = need_transform(tensor->type);
    ਤਾਰਕਿਕ is_nz        = !is_quantized && tensor->type != GGML_TYPE_BF16 && ਭਾਰ_ਮੰਜ਼ਿਲ_nz &&
                 is_matmul_ਭਾਰ((ਅਚਲ ggml_tensor *) tensor);

    // Plain tensor (not quantized, not NZ): direct copy, no tracking needed
    ਜੇ (!is_quantized && !is_nz) {
        ACL_CHECK(aclrtMemcpy((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size, ਅੰਕੜੇ, size, ACL_MEMCPY_HOST_ਮੰਜ਼ਿਲ_DEVICE));
        ਵਾਪਸ;
    }

    // Single-shot write (full tensor at once): handle directly without tracking overhead
    ਜੇ (ਵਿੱਥ == 0 && size == ggml_nbytes(tensor)) {
        ਜੇ (is_quantized) {
            ਖਾਲੀ * transform_ਬਫਰ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(size);
            ggml_backend_cann_transform(tensor, ਅੰਕੜੇ, transform_ਬਫਰ);
            ACL_CHECK(aclrtMemcpy(tensor->ਅੰਕੜੇ, size, transform_ਬਫਰ, size, ACL_MEMCPY_HOST_ਮੰਜ਼ਿਲ_DEVICE));
            ਮੁਕਤ(transform_ਬਫਰ);
        } ਨਹੀਂਤਾਂ {
            // NZ weight
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ACL_CHECK(aclrtMemcpy(tensor->ਅੰਕੜੇ, size, ਅੰਕੜੇ, size, ACL_MEMCPY_HOST_ਮੰਜ਼ਿਲ_DEVICE));
            ਭਾਰ_format_ਮੰਜ਼ਿਲ_nz(tensor, ctx->device);
        }
        ਵਾਪਸ;
    }

    // Chunked write: use tracker to accumulate progress and defer transform/conversion
    Tensor_ਸੈੱਟ_Tracker * tracker = ctx->get_or_create_tracker(tensor);
    std::unique_lock<std::mutex> ਤਾਲਾ(tracker->mtx);

    ਜੇ (is_quantized) {
        // Stage data in host buffer; transform requires full tensor data
        ਜੇ (tracker->host_ਬਫਰ.ਖਾਲੀ_ਥਾਵਾਂ()) {
            tracker->host_ਬਫਰ.resize(tracker->ਕੁੱਲ_ਬਾਈਟਾਂ);
        }
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(tracker->host_ਬਫਰ.ਅੰਕੜੇ() + ਵਿੱਥ, ਅੰਕੜੇ, size);
    } ਨਹੀਂਤਾਂ {
        // NZ weight: upload chunk to device immediately, defer conversion
        ACL_CHECK(aclrtMemcpy((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size, ਅੰਕੜੇ, size, ACL_MEMCPY_HOST_ਮੰਜ਼ਿਲ_DEVICE));
    }

    tracker->ਬਾਈਟਾਂ_written += size;

    // All chunks received: perform deferred transform/conversion
    ਜੇ (tracker->ਬਾਈਟਾਂ_written >= tracker->ਕੁੱਲ_ਬਾਈਟਾਂ) {
        ਜੇ (is_quantized) {
            ਖਾਲੀ * transform_ਬਫਰ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(tracker->ਕੁੱਲ_ਬਾਈਟਾਂ);
            ggml_backend_cann_transform(tensor, tracker->host_ਬਫਰ.ਅੰਕੜੇ(), transform_ਬਫਰ);
            ACL_CHECK(aclrtMemcpy(tensor->ਅੰਕੜੇ, tracker->ਕੁੱਲ_ਬਾਈਟਾਂ, transform_ਬਫਰ, tracker->ਕੁੱਲ_ਬਾਈਟਾਂ, ACL_MEMCPY_HOST_ਮੰਜ਼ਿਲ_DEVICE));
            ਮੁਕਤ(transform_ਬਫਰ);
        }

        ਜੇ (is_nz) {
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ਭਾਰ_format_ਮੰਜ਼ਿਲ_nz(tensor, ctx->device);
        }

        // Unlock before removing tracker, as remove_tracker destroys the mutex
        ਤਾਲਾ.unlock();
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                ਅਚਲ ggml_tensor *   tensor,
                                                ਖਾਲੀ *                ਅੰਕੜੇ,
                                                size_t                ਵਿੱਥ,
                                                size_t                size) {
    ggml_backend_cann_ਬਫਰ_context * ctx = (ggml_backend_cann_ਬਫਰ_context *) ਬਫਰ->context;

    ggml_cann_ਸੈੱਟ_device(ctx->device);

    ਜੇ (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(ਅੰਕੜੇ, size, (ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size, ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_HOST));
    } ਨਹੀਂਤਾਂ {
        ਖਾਲੀ * transform_ਬਫਰ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(size);
        ACL_CHECK(aclrtMemcpy(transform_ਬਫਰ, size, (ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size, ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_HOST));
        ggml_backend_cann_transform_ਪਿੱਠ(tensor, transform_ਬਫਰ, ਅੰਕੜੇ);
        ਮੁਕਤ(transform_ਬਫਰ);
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
ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cann_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                ਅਚਲ ggml_tensor *   src,
                                                ggml_tensor *         dst) {
    ਜੇ (ggml_backend_buft_is_cann(src->ਬਫਰ->buft)) {
        ggml_backend_cann_ਬਫਰ_context * src_ctx = (ggml_backend_cann_ਬਫਰ_context *) src->ਬਫਰ->context;
        ggml_backend_cann_ਬਫਰ_context * dst_ctx = (ggml_backend_cann_ਬਫਰ_context *) ਬਫਰ->context;

        size_t memcpy_ਮਾਪ = ggml_nbytes(src);
        // Same device.
        ਜੇ (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((ਅੱਖਰ *) dst->ਅੰਕੜੇ, memcpy_ਮਾਪ, (ਅਚਲ ਅੱਖਰ *) src->ਅੰਕੜੇ, memcpy_ਮਾਪ,
                                  ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_DEVICE));
            ਵਾਪਸ ਸੱਚ;
        } ਨਹੀਂਤਾਂ {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
            // TODO: Support 310p P2P copy
            ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device, dst_ctx->device));
            ਜੇ (canAccessPeer) {
                ggml_cann_ਸੈੱਟ_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((ਅੱਖਰ *) dst->ਅੰਕੜੇ, memcpy_ਮਾਪ, (ਅਚਲ ਅੱਖਰ *) src->ਅੰਕੜੇ, memcpy_ਮਾਪ,
                                      ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_DEVICE));
                ਵਾਪਸ ਸੱਚ;
            }
        }
    }
    ਵਾਪਸ ਝੂਠ;
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    ggml_backend_cann_ਬਫਰ_context * ctx = (ggml_backend_cann_ਬਫਰ_context *) ਬਫਰ->context;

    ggml_cann_ਸੈੱਟ_device(ctx->device);
    ACL_CHECK(aclrtMemset((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size, value, size));
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ggml_backend_cann_ਬਫਰ_context * ctx = (ggml_backend_cann_ਬਫਰ_context *) ਬਫਰ->context;

    ggml_cann_ਸੈੱਟ_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, ਬਫਰ->size, value, ਬਫਰ->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
ਸਥਿਰ ਅਚਲ ggml_backend_ਬਫਰ_i ggml_backend_cann_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_cann_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_cann_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_cann_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cann_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cann_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_cann_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ggml_backend_cann_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
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
ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_cann_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ggml_backend_cann_ਬਫਰ_type_context * buft_ctx = (ggml_backend_cann_ਬਫਰ_type_context *) buft->context;

    ggml_cann_ਸੈੱਟ_device(buft_ctx->device);

    ਅਚਲ size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    ਜੇ (size == 0) {
        size = alignment;
    }
    ਖਾਲੀ *   dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_ਪਹਿਲਾ);
    ਜੇ (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, buft_ctx->device, acl_Get_Recent_Err_ਸੁਨੇਹਾ());
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ggml_backend_cann_ਬਫਰ_context * ctx = ਨਵਾਂ ggml_backend_cann_ਬਫਰ_context(buft_ctx->device, dev_ptr);

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_cann_ਬਫਰ_interface, ctx, size);
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
ਸਥਿਰ size_t ggml_backend_cann_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ 128;

    GGML_ਅਣਵਰਤਿਆ(buft);
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
ਸਥਿਰ size_t ggml_backend_cann_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft,
                                                           ਅਚਲ ggml_tensor *        tensor) {
    size_t  size = ggml_nbytes(tensor);
    int64_t ne0  = tensor->ne[0];

    // Only check env once.
    ਸਥਿਰ ਤਾਰਕਿਕ ਭਾਰ_ਮੰਜ਼ਿਲ_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);
    ਜੇ (ggml_is_quantized(tensor->type)) {
        ਜੇ (ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING != 0) {
            size += ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING - ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);
        }
    } ਨਹੀਂਤਾਂ ਜੇ (ਭਾਰ_ਮੰਜ਼ਿਲ_nz && tensor->type != GGML_TYPE_BF16
               && is_matmul_ਭਾਰ((ਅਚਲ ggml_tensor *) tensor)) {
        // NZ format weight are not support quantized yet.
        // If ND tensor transform to NZ, size may changed.
        int64_t shape[] = { tensor->ne[1], tensor->ne[0] };
        GGML_ASSERT(tensor->ne[2] == 1);
        GGML_ASSERT(tensor->ne[3] == 1);
        ਅਚਲ aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
        size_t              new_ਮਾਪ;
        ACL_CHECK(aclnn_Calculate_Matmul_ਭਾਰ_ਮਾਪ_V2(acl_shape, ggml_cann_type_mapping(tensor->type), &new_ਮਾਪ));
        ACL_CHECK(aclDestroyIntArray(acl_shape));
        size = std::max(size, new_ਮਾਪ);
    }

    ਵਾਪਸ size;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cann_ਬਫਰ_type_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ਝੂਠ;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ਸਥਿਰ ਅਚਲ ggml_backend_ਬਫਰ_type_i ggml_backend_cann_ਬਫਰ_type_interface = {
    /* .get_name            = */ ggml_backend_cann_ਬਫਰ_type_name,
    /* .alloc_buffer        = */ ggml_backend_cann_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ਨਿਰਾ,
    /* .get_alignment       = */ ggml_backend_cann_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ਨਿਰਾ,  // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cann_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ggml_backend_cann_ਬਫਰ_type_is_host,
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
ggml_backend_ਬਫਰ_type_t ggml_backend_cann_ਬਫਰ_type(int32_t device) {
    ਸਥਿਰ std::mutex           mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

    ਜੇ (device >= ggml_backend_cann_get_device_ਗਿਣਤੀ()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਸਥਿਰ ggml_backend_ਬਫਰ_type ggml_backend_cann_ਬਫਰ_types[GGML_CANN_MAX_DEVICES];

    ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cann_ਬਫਰ_type_initialized = ਝੂਠ;

    ਜੇ (!ggml_backend_cann_ਬਫਰ_type_initialized) {
        ਲਈ (int32_t i = 0; i < ggml_cann_info().device_ਗਿਣਤੀ; i++) {
            ggml_backend_cann_ਬਫਰ_types[i] = {
                /* .iface    = */ ggml_backend_cann_ਬਫਰ_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), i),
                /* .context  = */
                ਨਵਾਂ ggml_backend_cann_ਬਫਰ_type_context{ i, "CANN" + std::to_string(i) },
            };
        }
        ggml_backend_cann_ਬਫਰ_type_initialized = ਸੱਚ;
    }

    ਵਾਪਸ &ggml_backend_cann_ਬਫਰ_types[device];
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
ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cann_host_ਬਫਰ_type_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ "CANN_Host";

    GGML_ਅਣਵਰਤਿਆ(buft);
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
ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cann_host_ਬਫਰ_name(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ "CANN_Host";

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer, including
 * its context.
 *
 * @param buffer The CANN host buffer to free.
 */
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_host_ਬਫਰ_free(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ACL_CHECK(aclrtFreeHost(ਬਫਰ->context));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
ਸਥਿਰ ਖਾਲੀ * ggml_cann_host_malloc(size_t size) {
    ਜੇ (getenv("GGML_CANN_NO_PINNED") != ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਅਚਲ size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    ਜੇ (size == 0) {
        size = alignment;
    }

    ਖਾਲੀ *   hostPtr = ਖਾਲੀ_ਸੂਚਕ;
    aclError err     = aclrtMallocHost((ਖਾਲੀ **) &hostPtr, size);
    ਜੇ (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                      acl_Get_Recent_Err_ਸੁਨੇਹਾ());
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_cann_host_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft,
                                                                             size_t                     size) {
    ਖਾਲੀ * hostPtr = ggml_cann_host_malloc(size);

    ਜੇ (hostPtr == ਖਾਲੀ_ਸੂਚਕ) {
        // fallback to cpu buffer
        ਵਾਪਸ ggml_backend_buft_alloc_ਬਫਰ(ggml_backend_cpu_ਬਫਰ_type(), size);
    }

    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_cpu_ਬਫਰ_from_ptr(hostPtr, size);
    ਬਫਰ->buft                 = buft;
    ਬਫਰ->iface.free_ਬਫਰ    = ggml_backend_cann_host_ਬਫਰ_free;

    ਵਾਪਸ ਬਫਰ;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_ਬਫਰ_type_t ggml_backend_cann_host_ਬਫਰ_type() {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_cann_ਬਫਰ_type_host = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cann_host_ਬਫਰ_type_name,
                           /* .alloc_buffer         = */ ggml_backend_cann_host_ਬਫਰ_type_alloc_ਬਫਰ,
                           /* .alloc_buffer_n       = */ ਨਿਰਾ,
                           /* .get_alignment        = */ ggml_backend_cpu_ਬਫਰ_type()->iface.get_alignment,
                           /* .get_max_size         = */ ਨਿਰਾ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_ਬਫਰ_type()->iface.get_alloc_ਮਾਪ,
                           /* .get_alloc_size_n     = */ ਨਿਰਾ,
                           /* .is_host              = */ ggml_backend_cpu_ਬਫਰ_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
        /* .context  = */ ਖਾਲੀ_ਸੂਚਕ,
    };

    ਵਾਪਸ &ggml_backend_cann_ਬਫਰ_type_host;
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
ਸਥਿਰ ਤਾਰਕਿਕ ggml_cann_compute_forward(ggml_backend_cann_context & ctx, ਬਣਤਰ ggml_tensor * dst) {
    ਚੋਣ (dst->op) {
        ਹਾਲਤ GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ggml_cann_get_ਪੰਕਤੀਆਂ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            ggml_cann_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_ADD1:
            ggml_cann_ਦੁਆਧਾਰੀ_op<aclnn_ਜੋੜੋ>(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SUB:
            ggml_cann_ਦੁਆਧਾਰੀ_op<aclnn_sub>(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL:
            ggml_cann_ਦੁਆਧਾਰੀ_op<aclnn_mul>(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIV:
            ggml_cann_ਦੁਆਧਾਰੀ_op<aclnn_div>(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(dst)) {
                ਹਾਲਤ GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_OP_UNARY(Abs);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_OP_UNARY(Neg);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_GELU:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                    // aclnnGelu internally uses the erf-based approximation.
                    GGML_CANN_CALL_OP_UNARY(Gelu);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_OP_UNARY(Silu);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                    {
                        ਸਵੈਚਾਲਿਤ lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                        };
                        ggml_cann_op_unary(lambda, ctx, dst);
                    }
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_TANH:
                    GGML_CANN_CALL_OP_UNARY(Tanh);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_RELU:
                    GGML_CANN_CALL_OP_UNARY(Relu);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Sigmoid);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Hardsigmoid);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                    GGML_CANN_CALL_OP_UNARY(Hardswish);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_EXP:
                    GGML_CANN_CALL_OP_UNARY(Exp);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ELU:
                    ggml_cann_elu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SGN:
                    GGML_CANN_CALL_OP_UNARY(Sign);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                    ggml_cann_ਕਦਮ(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                    ggml_cann_softplus(ctx, dst);
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(dst)) {
                ਹਾਲਤ GGML_GLU_OP_REGLU:
                    GGML_CANN_CALL_OP_UNARY_GATED(Relu);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                    ggml_cann_geglu(ctx, dst, 0);  // approximate=0 → tanh
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                    ggml_cann_geglu(ctx, dst, 1);  // approximate=1 → erf
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                    ggml_cann_swiglu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cann_swiglu_clamp(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                    ggml_cann_geglu_ਤੇਜ਼(ctx, dst);
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            ggml_cann_ਸਮੂਹ_norm(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_L2_NORM:
            ggml_cann_l2_norm(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cann_cross_entropy_loss(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == ਖਾਲੀ_ਸੂਚਕ);
            dst->src[1] = dst->src[0];
            ggml_cann_ਦੁਆਧਾਰੀ_op<aclnn_mul>(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQRT:
            GGML_CANN_CALL_OP_UNARY(Sqrt);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            ggml_cann_ਸੈੱਟ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
            ggml_cann_diag_ਨਕਾਬ(ctx, dst, -INFINITY);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            ggml_cann_im2col(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ:
            ggml_cann_ਜੋੜ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            ggml_cann_ਜੋੜ_ਪੰਕਤੀਆਂ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGMAX:
            ggml_cann_argmax(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COS:
            ggml_cann_op_unary<aclnn_cos>(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SIN:
            ggml_cann_op_unary<aclnn_sin>(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cann_conv_transpose_1d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LOG:
            GGML_CANN_CALL_OP_UNARY(Log);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            ggml_cann_ਔਸਤ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
            ggml_cann_pad_reflect_1d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
            ggml_cann_ਗਿਣਤੀ_equal(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
            ggml_cann_ਬਾਹਰਲਾ_prod(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
            ggml_cann_gated_ਰੇਖਿਕ_attn(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_CONV:
            ggml_cann_ssm_conv(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CUMSUM:
            ggml_cann_cumsum(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TRI:
            ggml_cann_tri(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FILL:
            ggml_cann_fill(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG:
            ggml_cann_diag(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            ggml_cann_solve_tri(ctx, dst);
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
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
ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;

    ਵਾਪਸ cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    ਮਿਟਾਓ cann_ctx;
    ਮਿਟਾਓ backend;
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_ਸੈੱਟ_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *  tensor,
                                               ਅਚਲ ਖਾਲੀ *   ਅੰਕੜੇ,
                                               size_t         ਵਿੱਥ,
                                               size_t         size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_ਬਫਰ_t       buf      = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;

    GGML_ASSERT(buf->buft == ggml_backend_cann_ਬਫਰ_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size, ਅੰਕੜੇ, size, ACL_MEMCPY_HOST_ਮੰਜ਼ਿਲ_DEVICE,
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_get_tensor_async(ggml_backend_t      backend,
                                               ਅਚਲ ggml_tensor * tensor,
                                               ਖਾਲੀ *              ਅੰਕੜੇ,
                                               size_t              ਵਿੱਥ,
                                               size_t              size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_ਬਫਰ_t       buf      = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;

    GGML_ASSERT(buf->buft == ggml_backend_cann_ਬਫਰ_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync(ਅੰਕੜੇ, size, (ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size, ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_HOST,
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
ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cann_cpy_tensor_async(ggml_backend_t      backend_src,
                                               ggml_backend_t      backend_dst,
                                               ਅਚਲ ggml_tensor * src,
                                               ggml_tensor *       dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) || ggml_backend_is_cann(backend_dst));

    GGML_ASSERT(!is_matmul_ਭਾਰ((ਅਚਲ ggml_tensor *) src));

    ਜੇ (!ggml_backend_buft_is_cann(src->ਬਫਰ->buft) || !ggml_backend_buft_is_cann(dst->ਬਫਰ->buft)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_backend_ਬਫਰ_t buf_src = src->view_src ? src->view_src->ਬਫਰ : src->ਬਫਰ;
    ggml_backend_ਬਫਰ_t buf_dst = dst->view_src ? dst->view_src->ਬਫਰ : dst->ਬਫਰ;

    ggml_backend_cann_context * cann_ctx_src = (ggml_backend_cann_context *) backend_src->context;
    ggml_backend_cann_context * cann_ctx_dst = (ggml_backend_cann_context *) backend_dst->context;

    size_t copy_ਮਾਪ = ggml_nbytes(dst);
    ਜੇ (copy_ਮਾਪ == 0) {
        ਵਾਪਸ ਸੱਚ;
    }
    ਜੇ (backend_src != backend_dst) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
        // TODO: Support 310p P2P copy
        ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
        ggml_backend_cann_ਬਫਰ_context * buf_ctx_src = (ggml_backend_cann_ਬਫਰ_context *) buf_src->context;
        ggml_backend_cann_ਬਫਰ_context * buf_ctx_dst = (ggml_backend_cann_ਬਫਰ_context *) buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device, cann_ctx_dst->device));
        ਜੇ (!canAccessPeer) {
            ਵਾਪਸ ਝੂਠ;
        }

        // need open both directions for memcpyasync between devices.
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_ਸੈੱਟ_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        // wait for task_queue empty to keep task order.
        ACL_CHECK(aclrtMemcpyAsync(dst->ਅੰਕੜੇ, copy_ਮਾਪ, src->ਅੰਕੜੇ, copy_ਮਾਪ, ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_DEVICE,
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
        ACL_CHECK(aclrt_Synchronize_ਧਾਰਾ(cann_ctx_src->stream()));
    } ਨਹੀਂਤਾਂ {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->ਅੰਕੜੇ, copy_ਮਾਪ, src->ਅੰਕੜੇ, copy_ਮਾਪ, ACL_MEMCPY_DEVICE_ਮੰਜ਼ਿਲ_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    ਵਾਪਸ ਸੱਚ;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_ਸੈੱਟ_device(cann_ctx->device);
    ACL_CHECK(aclrt_Synchronize_ਧਾਰਾ(cann_ctx->stream()));
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
ਸਥਿਰ ਤਾਰਕਿਕ ggml_cann_can_fuse(ਅਚਲ ਬਣਤਰ ggml_cgraph *          cgraph,
                               ਪੂਰਨ                                 ਗੰਢ_idx,
                               std::initializer_list<ਗਿਣਤੀ ggml_op> ops) {
    ਜੇ (!ggml_can_fuse(cgraph, ਗੰਢ_idx, ops)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // CANN backend supports fusing ADD + RMS_NORM operations
    ਜੇ ((ops.size() == 2) && ops.begin()[0] == GGML_OP_ਜੋੜੋ && ops.begin()[1] == GGML_OP_RMS_NORM) {
        ggml_tensor * ਜੋੜੋ_ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];
        // TODO: support broadcast for ADD + RMS_NORM
        ਜੇ (ਜੋੜੋ_ਗੰਢ->src[0]->ne[0] != ਜੋੜੋ_ਗੰਢ->src[1]->ne[0] || ਜੋੜੋ_ਗੰਢ->src[0]->ne[1] != ਜੋੜੋ_ਗੰਢ->src[1]->ne[1] ||
            ਜੋੜੋ_ਗੰਢ->src[0]->ne[2] != ਜੋੜੋ_ਗੰਢ->src[1]->ne[2] || ਜੋੜੋ_ਗੰਢ->src[0]->ne[3] != ਜੋੜੋ_ਗੰਢ->src[1]->ne[3]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ ਸੱਚ;
    }

    ਵਾਪਸ ਝੂਠ;
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
ਸਥਿਰ ਖਾਲੀ evaluate_and_capture_cann_ਗ੍ਰਾਫ(ggml_backend_cann_context * cann_ctx,
                                            ggml_cgraph *               cgraph,
                                            ਤਾਰਕਿਕ                        use_cann_ਗ੍ਰਾਫ,
                                            ਤਾਰਕਿਕ                        cann_ਗ੍ਰਾਫ_capture_required) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ USE_ACL_ਗ੍ਰਾਫ
    ਜੇ (use_cann_ਗ੍ਰਾਫ && cann_ਗ੍ਰਾਫ_capture_required) {  // Begin CANN graph capture
        ACL_CHECK(aclmdlRICaptureBegin(cann_ctx->stream(), ACL_MODEL_RI_CAPTURE_ਬਹੁਲਕ_GLOBAL));
    }
#ਜੇ_ਅੰਤ  // USE_ACL_GRAPH
    // Only perform the graph execution if CANN graphs are not enabled, or we are capturing the graph.
    // With the use of CANN graphs, the execution will be performed by the graph launch.
    ਸਥਿਰ ਤਾਰਕਿਕ opt_fusion = parse_bool(get_env_as_lowercase("GGML_CANN_OPERATOR_FUSION").value_or(""));

    ਜੇ (!use_cann_ਗ੍ਰਾਫ || cann_ਗ੍ਰਾਫ_capture_required) {
        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * ਗੰਢ = cgraph->nodes[i];
            ਜੇ (opt_fusion) {
                ਜੇ (ggml_cann_can_fuse(cgraph, i, { GGML_OP_ਜੋੜੋ, GGML_OP_RMS_NORM })) {
                    ggml_cann_op_ਜੋੜੋ_rms_norm_fused(*cann_ctx, ਗੰਢ, cgraph->nodes[i + 1]);
                    i++;
                    ਜਾਰੀ;
                }
            }

            ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗੰਢ) || ਗੰਢ->op == GGML_OP_RESHAPE || ਗੰਢ->op == GGML_OP_TRANSPOSE ||
                ਗੰਢ->op == GGML_OP_VIEW || ਗੰਢ->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ || ਗੰਢ->op == GGML_OP_NONE) {
                ਜਾਰੀ;
            }

            ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ਜਾਰੀ;
            }

            ਤਾਰਕਿਕ ok = ggml_cann_compute_forward(*cann_ctx, ਗੰਢ);
            ਜੇ (!ok) {
                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, ਗੰਢ->name, ggml_op_name(ਗੰਢ->op));
            }
            GGML_ASSERT(ok);
        }
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ USE_ACL_ਗ੍ਰਾਫ
    ਜੇ (use_cann_ਗ੍ਰਾਫ) {
        GGML_ASSERT(!cann_ctx->ਗ੍ਰਾਫ_lru_cache.cache_list.ਖਾਲੀ_ਥਾਵਾਂ());
        ggml_cann_ਗ੍ਰਾਫ * matched_ਗ੍ਰਾਫ = cann_ctx->ਗ੍ਰਾਫ_lru_cache.cache_list.front();

        ਜੇ (cann_ਗ੍ਰਾਫ_capture_required) {  // End CANN graph capture
            ACL_CHECK(aclmdl_RICapture_ਸਿਰਾ(cann_ctx->stream(), &matched_ਗ੍ਰਾਫ->graph));
        }

        // Execute CANN graph
        ACL_CHECK(aclmdlRIExecuteAsync(matched_ਗ੍ਰਾਫ->graph, cann_ctx->stream()));
    }
#ਜੇ_ਅੰਤ  // USE_ACL_GRAPH
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
ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_cann_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_ਸੈੱਟ_device(cann_ctx->device);
    g_nz_workspaces[cann_ctx->device].clear();

    // calculate rope cache for fist layer in current device.
    cann_ctx->rope_cache.cached = ਝੂਠ;

    ਤਾਰਕਿਕ ਗ੍ਰਾਫ_capture_required = ਝੂਠ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ USE_ACL_ਗ੍ਰਾਫ
    ਤਾਰਕਿਕ use_cann_ਗ੍ਰਾਫ = ਸੱਚ;

    ਸਥਿਰ ਤਾਰਕਿਕ prefill_use_ਗ੍ਰਾਫ = parse_bool(get_env_as_lowercase("GGML_CANN_PREFILL_USE_GRAPH").value_or(""));
    ਜੇ (!prefill_use_ਗ੍ਰਾਫ) {
        // Do not use acl_graph for prefill.
        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * ਗੰਢ = cgraph->nodes[i];
            // TODO: Optimize here. Currently, we can only
            // get seq_len by FA's input.
            ਜੇ (ਗੰਢ->op == GGML_OP_FLASH_ATTN_EXT) {
                // Q -> src[0], shape: [B, S, N, D]
                use_cann_ਗ੍ਰਾਫ = (ਗੰਢ->src[0]->ne[1] == 1);
                ਰੋਕੋ;
            }
        }
    }

    ਜੇ (!cann_ctx->acl_ਗ੍ਰਾਫ_ਬਹੁਲਕ) {
        use_cann_ਗ੍ਰਾਫ = ਝੂਠ;
    }

    ਜੇ (use_cann_ਗ੍ਰਾਫ) {
        // If no matching graph is found, the graph needs to be recaptured.
        ਗ੍ਰਾਫ_capture_required = !cann_ctx->ਗ੍ਰਾਫ_lru_cache.ਲੱਭੋ_and_move_ਮੰਜ਼ਿਲ_front(cgraph);

        ਜੇ (ਗ੍ਰਾਫ_capture_required) {
            // If no matching graph is found, add a new ACL graph.
            ggml_cann_ਗ੍ਰਾਫ * new_ਗ੍ਰਾਫ = ggml_cann_ਗ੍ਰਾਫ::create_from_cgraph(cgraph);
            cann_ctx->ਗ੍ਰਾਫ_lru_cache.ਧੱਕੋ(new_ਗ੍ਰਾਫ);

            // Pre-load rope cache before graph capture.  During capture the
            // stream cannot perform host-to-device memcpy or device memory
            // malloc/free.  Running the full cache init now populates the
            // cache metadata so these branches are skipped during capture,
            // while also warming up the memory pool.
            ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ਗੰਢ = cgraph->nodes[i];
                ਜੇ (ਗੰਢ->op == GGML_OP_ROPE) {
                    ggml_cann_rope_cache_preload(*cann_ctx, ਗੰਢ);
                    ਰੋਕੋ;
                }
            }
        }
    }
#ਨਹੀਂਤਾਂ
    ਤਾਰਕਿਕ use_cann_ਗ੍ਰਾਫ = ਝੂਠ;
#ਜੇ_ਅੰਤ  // USE_ACL_GRAPH
    evaluate_and_capture_cann_ਗ੍ਰਾਫ(cann_ctx, cgraph, use_cann_ਗ੍ਰਾਫ, ਗ੍ਰਾਫ_capture_required);

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
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
ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cann_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(op)) {
                ਹਾਲਤ GGML_UNARY_OP_ABS:
                ਹਾਲਤ GGML_UNARY_OP_NEG:
                ਹਾਲਤ GGML_UNARY_OP_GELU:
                ਹਾਲਤ GGML_UNARY_OP_SILU:
                ਹਾਲਤ GGML_UNARY_OP_RELU:
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                ਹਾਲਤ GGML_UNARY_OP_TANH:
                ਹਾਲਤ GGML_UNARY_OP_EXP:
                ਹਾਲਤ GGML_UNARY_OP_ELU:
                ਹਾਲਤ GGML_UNARY_OP_SGN:
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                    ਵਾਪਸ ਸੱਚ;
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(op)) {
                ਹਾਲਤ GGML_GLU_OP_REGLU:
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                    ਵਾਪਸ ਸੱਚ;
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                ਚੋਣ (op->src[0]->type) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                    ਹਾਲਤ GGML_TYPE_BF16:
#ਜੇ_ਅੰਤ
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_F32:
                        ਵਾਪਸ ਸੱਚ;
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                        // Q4 && Q8 per group is not support on 310p device
                        ਵਾਪਸ ਝੂਠ;
#ਨਹੀਂਤਾਂ
                        // only support contiguous for quantized types.
                        ਵਾਪਸ ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#ਜੇ_ਅੰਤ
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
            }
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            ਚੋਣ (op->src[0]->type) {
                ਹਾਲਤ GGML_TYPE_F16:
                ਹਾਲਤ GGML_TYPE_F32:
                    ਵਾਪਸ ਸੱਚ;
                ਹਾਲਤ GGML_TYPE_Q8_0:
                ਹਾਲਤ GGML_TYPE_Q4_0:
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                    // Q4 && Q8 per group is not support on 310p device
                    ਵਾਪਸ ਝੂਠ;
#ਨਹੀਂਤਾਂ
                    // only support contiguous for quantized types.
                    ਵਾਪਸ ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#ਜੇ_ਅੰਤ
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
        // embedding
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            {
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                    ਹਾਲਤ GGML_TYPE_BF16:
#ਜੇ_ਅੰਤ
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            {
                ਚੋਣ (op->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                    ਹਾਲਤ GGML_TYPE_BF16:
#ਜੇ_ਅੰਤ
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CPY:
            {
                ggml_tensor * src = op->src[0];
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                ਜੇ ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                    // only support F32 and F16 on 310P.
                    ਵਾਪਸ ਝੂਠ;
                }
#ਨਹੀਂਤਾਂ
                ਜੇ ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
                    // only support F32, F16 and BF16.
                    ਵਾਪਸ ਝੂਠ;
                }
#ਜੇ_ਅੰਤ
                ਵਾਪਸ ਸੱਚ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONT:
            {
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_F16:
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                    ਹਾਲਤ GGML_TYPE_BF16:
#ਜੇ_ਅੰਤ
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
            }
        ਹਾਲਤ GGML_OP_ROPE:
            {
                ਜੇ (((ਅਚਲ int32_t *) op->op_params)[15] != 0) {
                    ਵਾਪਸ ਝੂਠ; // FIXME: support ggml_rope_set_offset
                }
                ਜੇ (op->src[0]->ne[0] > 896) {
                    ਵਾਪਸ ਝੂਠ;
                }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                // TODO: Support rope_dim < ne00(dim)
                ਜੇ (op->src[0]->ne[0] != op->op_params[1]) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (!ggml_is_contiguous(op->src[0])) {
                    ਵਾਪਸ ਝੂਠ;
                }
#ਜੇ_ਅੰਤ
                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_UPSCALE:
            {
                // aclnnUpsampleNearest2dGetWorkspaceSize not support
                // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
                ਜੇ (op->src[0]->ne[2] * op->ne[3] != op->src[0]->ne[3] * op->ne[2]) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->op_params[0] != GGML_SCALE_ਬਹੁਲਕ_ਸਭ_ਤੋਂ_ਨੇੜੇ) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_POOL_2D:
            {
                ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *) op->op_params;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                ਗਿਣਤੀ ggml_op_pool opt = static_cast<ggml_op_pool>(opts[0]);
                ਜੇ (opt == GGML_OP_POOL_MAX) {
                    ਵਾਪਸ ਝੂਠ;
                }
#ਜੇ_ਅੰਤ
                ਅਚਲ ਪੂਰਨ k0 = opts[1];
                ਅਚਲ ਪੂਰਨ k1 = opts[2];
                ਅਚਲ ਪੂਰਨ p0 = opts[5];
                ਅਚਲ ਪੂਰਨ p1 = opts[6];
                // value of paddingH should be at most half of kernelH
                // value of paddingW should be at most half of kernelW
                ਵਾਪਸ (p0 <= (k0 / 2)) && (p1 <= (k1 / 2));
            }
        ਹਾਲਤ GGML_OP_ਜੋੜ:
            ਵਾਪਸ ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]);
        ਹਾਲਤ GGML_OP_L2_NORM:
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
        ਹਾਲਤ GGML_OP_DUP:
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
        ਹਾਲਤ GGML_OP_CONCAT:
        ਹਾਲਤ GGML_OP_REPEAT:
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_ADD1:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_DIV:
        ਹਾਲਤ GGML_OP_RMS_NORM:
        ਹਾਲਤ GGML_OP_SQR:
        ਹਾਲਤ GGML_OP_SQRT:
        ਹਾਲਤ GGML_OP_CLAMP:
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
        ਹਾਲਤ GGML_OP_ARGSORT:
        ਹਾਲਤ GGML_OP_ACC:
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_PAD:
            // TODO: add circular padding support for cann, see https://github.com/ggml-org/llama.cpp/pull/16985
            ਵਾਪਸ ggml_get_op_params_i32(op, 8) == 0;
        ਹਾਲਤ GGML_OP_ARANGE:
        ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
        ਹਾਲਤ GGML_OP_ARGMAX:
        ਹਾਲਤ GGML_OP_COS:
        ਹਾਲਤ GGML_OP_SIN:
        ਹਾਲਤ GGML_OP_LOG:
        ਹਾਲਤ GGML_OP_ਔਸਤ:
        ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
        ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
            {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                // Ger is not supported on 310p device
                ਵਾਪਸ ਝੂਠ;
#ਨਹੀਂਤਾਂ
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_F32:
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
#ਜੇ_ਅੰਤ
            }
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_SCALE:
            ਦਸ਼ਮਲਵ bias;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) (op->op_params) + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            ਵਾਪਸ bias == 0.0f;  // TODO: support bias != 0.0f
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            // TODO: support attention sinks [TAG_ATTN_SINKS]
            ਜੇ (op->src[2]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ASCEND_310P
                // FA not support on 310p device
                ਵਾਪਸ ਝੂਠ;
#ਨਹੀਂਤਾਂ
                // derived from [ggml-cuda.cu]
                ਜੇ (op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32 &&
                    op->src[1]->type != GGML_TYPE_BF16) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16) {
                    ਵਾਪਸ ਝੂਠ;
                }
                // TODO: support attention sinks [TAG_ATTN_SINKS]
                ਜੇ (op->src[4]) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਜੇ (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                    // different head sizes of K and V are not supported yet
                    ਵਾਪਸ ਝੂਠ;
                }
                ਦਸ਼ਮਲਵ logitSoftcap = 0.0f;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logitSoftcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) (op->op_params) + 2, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਜੇ (logitSoftcap != 0.0f) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਵਾਪਸ ਸੱਚ;
#ਜੇ_ਅੰਤ
            }
        ਹਾਲਤ GGML_OP_SSM_CONV:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_CUMSUM:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_TRI:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_FILL:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_DIAG:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    GGML_ਅਣਵਰਤਿਆ(dev);
}

/**
 * @brief Records an event on the CANN backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_event_ਰਿਕਾਰਡ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrt_ਰਿਕਾਰਡ_Event((aclrtEvent) event->context, cann_ctx->stream()));
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ਜੇ (ggml_backend_is_cann(backend)) {
        ACL_CHECK(aclrt_ਧਾਰਾ_Wait_Event(cann_ctx->stream(), (aclrtEvent) event->context));
    } ਨਹੀਂਤਾਂ {
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
ਸਥਿਰ ਅਚਲ ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_cann_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ggml_backend_cann_event_ਰਿਕਾਰਡ,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .graph_optimize          = */ ਨਿਰਾ,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
ਸਥਿਰ ggml_guid_t ggml_backend_cann_guid() {
    ਸਥਿਰ ggml_guid guid = { 0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                              0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64 };
    ਵਾਪਸ &guid;
}

// backend device
ਬਣਤਰ ggml_backend_cann_device_context {
    ਪੂਰਨ         device;
    std::string name;
    std::string description;
    ਪੂਰਨ op_offload_min_batch_ਮਾਪ;
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cann_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ਵਾਪਸ ctx->description.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ggml_backend_cann_get_device_ਸਮ੍ਰਿਤੀ(ctx->device, ਮੁਕਤ, ਕੁੱਲ);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_cann_device_get_type(ggml_backend_dev_t dev) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type        = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);

    ਤਾਰਕਿਕ host_ਬਫਰ = getenv("GGML_CANN_NO_PINNED") == ਖਾਲੀ_ਸੂਚਕ;

    props->caps = {
        /* .async                 = */ ਝੂਠ,
        /* .host_buffer           = */ host_ਬਫਰ,
        /* .buffer_from_host_ptr  = */ ਝੂਠ,
        /* .events                = */ ਸੱਚ,
        /* .mmap_support          = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    GGML_ਅਣਵਰਤਿਆ(params);
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ਵਾਪਸ ggml_backend_cann_init(ctx->device);
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
ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cann_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਜੇ (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context *      dev_ctx  = (ggml_backend_cann_device_context *) dev->context;
        ggml_backend_cann_ਬਫਰ_type_context * buft_ctx = (ggml_backend_cann_ਬਫਰ_type_context *) buft->context;
        ਵਾਪਸ buft_ctx->device == dev_ctx->device;
    }
    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_cann_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ਵਾਪਸ ggml_backend_cann_ਬਫਰ_type(ctx->device);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_cann_device_get_host_ਬਫਰ_type(ggml_backend_dev_t dev) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ਵਾਪਸ ggml_backend_cann_host_ਬਫਰ_type();
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
ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cann_offload_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;

    ਵਾਪਸ op->ne[1] >= dev_ctx->op_offload_min_batch_ਮਾਪ && op->op != GGML_OP_GET_ਪੰਕਤੀਆਂ;
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
ਸਥਿਰ ggml_backend_event_t ggml_backend_cann_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *) dev->context;

    ggml_cann_ਸੈੱਟ_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    ਵਾਪਸ ਨਵਾਂ ggml_backend_event{
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
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent) event->context));

    ਮਿਟਾਓ event;
    GGML_ਅਣਵਰਤਿਆ(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
ਸਥਿਰ ਖਾਲੀ ggml_backend_cann_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent) event->context));

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,  // called for every card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type    = */ ggml_backend_cann_device_get_host_ਬਫਰ_type,
    /* .buffer_from_host_ptr    = */ ਨਿਰਾ,  // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};

// backend reg
ਬਣਤਰ ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_ਅਣਵਰਤਿਆ(reg);
    ਵਾਪਸ GGML_CANN_NAME;
}

ਸਥਿਰ size_t ggml_backend_cann_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    ਵਾਪਸ ctx->devices.size();
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    GGML_ASSERT(ਸੂਚਕਾਂਕ < ctx->devices.size());
    ਵਾਪਸ ctx->devices[ਸੂਚਕਾਂਕ];
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_cann_reg_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    GGML_ਅਣਵਰਤਿਆ(reg);
    GGML_ਅਣਵਰਤਿਆ(name);
    // reserved for future use
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

ਸਥਿਰ ਅਚਲ ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_ਗਿਣਤੀ,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_ਪਤਾ,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    ਸਥਿਰ ggml_backend_reg reg;
    ਸਥਿਰ ਤਾਰਕਿਕ             initialized = ਝੂਠ;

    {
        ਸਥਿਰ std::mutex           mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
        ਜੇ (!initialized) {
            aclInit(ਖਾਲੀ_ਸੂਚਕ);
            ggml_backend_cann_reg_context * ctx = ਨਵਾਂ ggml_backend_cann_reg_context;
            ਅਚਲ ਪੂਰਨ min_batch_ਮਾਪ = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            ਲਈ (ਪੂਰਨ i = 0; i < ggml_cann_info().device_ਗਿਣਤੀ; i++) {
                ggml_backend_cann_device_context * dev_ctx = ਨਵਾਂ ggml_backend_cann_device_context();
                dev_ctx->description                       = aclrtGetSocName();
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_CANN_NAME + std::to_string(i);
                dev_ctx->op_offload_min_batch_ਮਾਪ         = min_batch_ਮਾਪ;
                ggml_cann_ਸੈੱਟ_device(i);
                ggml_backend_dev_t dev = ਨਵਾਂ ggml_backend_device{ /* .iface   = */ ggml_backend_cann_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
                                    /* .iface       = */ ggml_backend_cann_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = ਸੱਚ;
    }

    ਵਾਪਸ &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(ਖਾਲੀ_ਸੂਚਕ);
    ਜੇ (device < 0 || device >= ggml_backend_cann_get_device_ਗਿਣਤੀ()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ggml_backend_cann_context * ctx = ਨਵਾਂ ggml_backend_cann_context(device);
    ਜੇ (ctx == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ggml_cann_ਸੈੱਟ_device(ctx->device);
    ggml_backend_t cann_backend =
        ਨਵਾਂ ggml_backend{ /* .guid      = */ ggml_backend_cann_guid(),
                          /* .interface = */ ggml_backend_cann_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                          /* .context   = */ ctx };

    ਵਾਪਸ cann_backend;
}

ਤਾਰਕਿਕ ggml_backend_is_cann(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_ਗਿਣਤੀ() {
    ਵਾਪਸ ggml_cann_info().device_ਗਿਣਤੀ;
}

ਖਾਲੀ ggml_backend_cann_get_device_description(int32_t device, ਅੱਖਰ * description, size_t description_ਮਾਪ) {
    ggml_cann_ਸੈੱਟ_device(device);
    ਅਚਲ ਅੱਖਰ * soc_name = aclrtGetSocName();
    snprintf(description, description_ਮਾਪ, "%s", soc_name);
}

ਖਾਲੀ ggml_backend_cann_get_device_ਸਮ੍ਰਿਤੀ(int32_t device, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ggml_cann_ਸੈੱਟ_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, ਮੁਕਤ, ਕੁੱਲ));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)

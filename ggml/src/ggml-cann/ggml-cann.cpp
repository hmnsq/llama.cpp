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

#అన్తర్భావ "ggml-cann.h"

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-cann/aclnn_ops.h"
#అన్తర్భావ "ggml-cann/common.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <acl/acl.h>
#అన్తర్భావ <aclnnop/aclnn_trans_matmul_weight.h>
#అన్తర్భావ <stdarg.h>

#అన్తర్భావ <chrono>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstring>
#అన్తర్భావ <memory>
#అన్తర్భావ <mutex>
#అన్తర్భావ <optional>
#అన్తర్భావ <queue>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <vector>

#నిర్వచించు GGML_COMMON_DECL_C

#అన్తర్భావ "ggml-common.h"

#నిర్వచించు GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] శూన్య ggml_cann_error(స్థిరరాశి అక్షరం * stmt, స్థిరరాశి అక్షరం * func, స్థిరరాశి అక్షరం * file, పూర్ణాంకం పంక్తి, స్థిరరాశి అక్షరం * సందేశం_2) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", సందేశం_2);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, పంక్తి);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

// Thread-local variable to record the current device of this thread.
thread_local పూర్ణాంకం g_current_cann_device = -1;

/**
 * @brief Set the CANN device to be used.
 *
 * @param device The target device ID to set.
 */
శూన్య ggml_cann_సమితి_device(స్థిరరాశి int32_t device) {
    // int current_device = -1;
    // Note: In some CANN versions, if no device has been set yet,
    //       aclrtGetDevice(&current_device) may return 0 by default.
    // aclrtGetDevice(&current_device);

    // If the current device is already the target one, no need to switch.
    ఒకవేళ (device == g_current_cann_device) {
        తిరిగి;
    }

    // Switch to the new device.
    ACL_CHECK(aclrt_సమితి_Device(device));

    // Update the global device record.
    g_current_cann_device = device;
}

/**
 * @brief Get the value of the specified environment variable (name) as lowercase.
 *        if not empty, return a std::string object
 */
std::optional<std::string> get_env_as_lowercase(స్థిరరాశి std::string & name) {
    స్థిరరాశి అక్షరం * val = std::getenv(name.c_str());
    ఒకవేళ (!val) {
        తిరిగి std::nullopt;
    }
    std::string res = std::string(val);
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    తిరిగి res;
}

/**
 * @brief Verify whether the environment variable is a valid value.
 */
తార్కిక parse_bool(స్థిరరాశి std::string & value) {
    స్థిర స్థిరరాశి std::unordered_set<std::string> valid_విలువలు = { "on", "1", "yes", "y", "enable", "true" };
    తిరిగి valid_విలువలు.find(value) != valid_విలువలు.end();
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
పూర్ణాంకం parse_integer(స్థిరరాశి std::string & value) {
    ప్రయత్నించు {
        తిరిగి std::stoi(value);
    } పట్టుకో (...) {
        తిరిగి 0;
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
స్థిర ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrt_Get_Device_లెక్క((uint32_t *) &info.device_లెక్క);

    ఒకవేళ (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n", __func__, acl_Get_Recent_Err_సందేశం());
        తిరిగి info;
    }

    GGML_ASSERT(info.device_లెక్క <= GGML_CANN_MAX_DEVICES);

    కొరకు (పూర్ణాంకం id = 0; id < info.device_లెక్క; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
        prop.కేటాయింపు_Type       = ACL_MEM_కేటాయింపు_TYPE_PINNED;
        prop.memAttr              = ACL_HBM_MEM_HUGE;
        prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = id;
        prop.reserve              = 0;
        err                       = aclrt_Mem_Get_కేటాయింపు_Granularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                                                                     &info.devices[id].vmm_granularity);
        info.devices[id].vmm      = err == ACL_SUCCESS;

        size_t విడుదల, మొత్తం;
        ggml_backend_cann_get_device_జ్ఞాపకం(id, &విడుదల, &మొత్తం);
        info.devices[id].మొత్తం_vram = విడుదల;
    }

    // TODO: add more device info later.
    తిరిగి info;
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
స్థిరరాశి ggml_cann_device_info & ggml_cann_info() {
    స్థిర ggml_cann_device_info info = ggml_cann_init();
    తిరిగి info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(priority segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
నిర్మాణం ggml_cann_pool_buf_prio : బహిరంగ ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    స్థిర స్థిరరాశి size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    స్థిర స్థిరరాశి size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    స్థిర స్థిరరాశి size_t alignment = 128;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    పూర్ణాంకం device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    తార్కిక disable_clean = అబద్ధం;

    /**
     * @brief Structure representing a CANN buffer.
     */
    నిర్మాణం ggml_cann_బఫరు {
        శూన్య *                                ptr  = శూన్యసూచిక;  ///< Pointer to the buffer.
        size_t                                size = 0;        ///< Size of the buffer.
        std::chrono::steady_clock::time_point చివరి_వాడిన;       ///< Last used time.

        తార్కిక క్రియాకారకం>(స్థిరరాశి ggml_cann_బఫరు & other) స్థిరరాశి { తిరిగి size > other.size; }
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    std::unordered_map<శూన్య *, size_t>                                                   బఫరు_pool;
    std::priority_queue<ggml_cann_బఫరు, std::vector<ggml_cann_బఫరు>, std::greater<>> free_buffers;

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_కొలత = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    స్పష్టమైన ggml_cann_pool_buf_prio(పూర్ణాంకం device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf_prio() {
        ggml_cann_సమితి_device(device);
        కొరకు (స్వయంచాలక & [b_ptr, b_కొలత] : బఫరు_pool) {
            aclrtFree(b_ptr);
            pool_కొలత -= b_కొలత;
        }
        బఫరు_pool.clear();
        GGML_ASSERT(pool_కొలత == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    శూన్య * alloc(size_t size, size_t * actual_కొలత) override {
        size = GGML_PAD(size, alignment);
        ఒకవేళ (size == 0) {
            size = alignment;
        }

        శూన్య * ptr = శూన్యసూచిక;
        స్వయంచాలక   now = std::chrono::steady_clock::now();

        std::vector<ggml_cann_బఫరు> free_buffers_rest;
        free_buffers_rest.reserve(free_buffers.size());
        అయితే (!free_buffers.ఖాళీలు()) {
            స్వయంచాలక b = free_buffers.top();
            free_buffers.తీయి();

            ఒకవేళ (b.size >= size) {
                // reuse the buffer if the size is enough
                స్థిరరాశి size_t margin = b.size - size;
                ఒకవేళ (margin <= max_reuse_margin) {
                    *actual_కొలత = b.size;
                    ptr          = b.ptr;
#నిర్వచితమైతే DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#అయితే_ముగింపు
                    ఆపు;
                }
            }

            తార్కిక should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.చివరి_వాడిన).count() > 100;
            ఒకవేళ (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_కొలత -= b.size;
                బఫరు_pool.erase(b.ptr);
#నిర్వచితమైతే DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#అయితే_ముగింపు
                కొనసాగు;
            }
            free_buffers_rest.push_back(b);
        }
        కొరకు (ggml_cann_బఫరు & b : free_buffers_rest) {
            free_buffers.నెట్టు(std::move(b));
        }

#నిర్వచితమైతే DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d] free pool_size = %5u MB\n\n", device,
                      (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576));
#అయితే_ముగింపు
        ఒకవేళ (ptr != శూన్యసూచిక) {
            తిరిగి ptr;
        }

        // allocate a new buffer if no buffer can be reused
        ggml_cann_సమితి_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_మొదటి));
        *actual_కొలత = size;
        pool_కొలత += size;
#నిర్వచితమైతే DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: allocate %p, "
            "pool_size = %5u MB, "
            "size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576),
            (uint32_t) (GGML_PAD(size, 1048576) / 1048576));
#అయితే_ముగింపు
        బఫరు_pool.emplace(ptr, size);
        తిరిగి ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    శూన్య విడుదల(శూన్య * ptr, size_t size) override {
        GGML_వాడని(size);
        స్వయంచాలక it = బఫరు_pool.find(ptr);
        ఒకవేళ (it == బఫరు_pool.end()) {
            GGML_ABORT("cann pool[%d]: buffer %p not found in pool\n", device, ptr);
        }

        స్వయంచాలక now = std::chrono::steady_clock::now();
        free_buffers.emplace(ggml_cann_బఫరు{ ptr, it->second, now });
#నిర్వచితమైతే DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: return   %p, "
            "pool_size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576));
#అయితే_ముగింపు
    }
};

/**
 * @brief A pool of CANN buffers(segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
నిర్మాణం ggml_cann_pool_buf : బహిరంగ ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    స్థిర స్థిరరాశి size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    స్థిర స్థిరరాశి size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    స్థిర స్థిరరాశి size_t alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    స్థిర స్థిరరాశి పూర్ణాంకం MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    పూర్ణాంకం device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    తార్కిక disable_clean = అబద్ధం;

    /**
     * @brief Structure representing a CANN buffer.
     */
    నిర్మాణం ggml_cann_బఫరు {
        శూన్య *                                ptr  = శూన్యసూచిక;  ///< Pointer to the buffer memory.
        size_t                                size = 0;        ///< Size of the buffer.
        తార్కిక                                  వాడిన = అబద్ధం;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point చివరి_వాడిన;       ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_బఫరు బఫరు_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_కొలత = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    స్పష్టమైన ggml_cann_pool_buf(పూర్ణాంకం device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf() {
        ggml_cann_సమితి_device(device);
        కొరకు (పూర్ణాంకం i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_బఫరు & b = బఫరు_pool[i];
            ఒకవేళ (b.ptr != శూన్యసూచిక) {
                aclrtFree(b.ptr);
                pool_కొలత -= b.size;
            }
        }
        GGML_ASSERT(pool_కొలత == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    శూన్య * alloc(size_t size, size_t * actual_కొలత) override {
        size = GGML_PAD(size, alignment);
        ఒకవేళ (size == 0) {
            size = alignment;
        }

        శూన్య * ptr = శూన్యసూచిక;
        స్వయంచాలక   now = std::chrono::steady_clock::now();

        పూర్ణాంకం i = 0;
        కొరకు (; i < MAX_BUFFERS; ++i) {
            ggml_cann_బఫరు & b = బఫరు_pool[i];
            ఒకవేళ (b.ptr == శూన్యసూచిక) {
                ఆపు;
            }
            ఒకవేళ (b.వాడిన) {
                కొనసాగు;
            }
            ఒకవేళ (b.size >= size) {
                // reuse the buffer if the size is enough
                స్థిరరాశి size_t margin = b.size - size;
                ఒకవేళ (margin <= max_reuse_margin) {
                    *actual_కొలత = b.size;
                    b.వాడిన       = నిజం;
                    ptr          = b.ptr;
#నిర్వచితమైతే DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#అయితే_ముగింపు
                    ఆపు;
                }
            }

            తార్కిక should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.చివరి_వాడిన).count() > 100;
            ఒకవేళ (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_కొలత -= b.size;
#నిర్వచితమైతే DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#అయితే_ముగింపు
                b.ptr = శూన్యసూచిక;
            }
        }
        ఒకవేళ (ptr != శూన్యసూచిక) {
            తిరిగి ptr;
        }

        ఒకవేళ (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_cann_బఫరు & b = బఫరు_pool[i];
            ggml_cann_సమితి_device(device);
            ACL_CHECK(aclrtMalloc(&b.ptr, size, ACL_MEM_MALLOC_HUGE_మొదటి));
            pool_కొలత += size;
            *actual_కొలత = size;
            b.size       = size;
            b.వాడిన       = నిజం;
            ఒకవేళ (i >= MAX_BUFFERS - 8) {
                GGML_LOG_WARN("cann pool[%d]: slots almost full\n", device);
            }
#నిర్వచితమైతే DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576),
                (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#అయితే_ముగింపు
            తిరిగి b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    శూన్య విడుదల(శూన్య * ptr, size_t size) override {
        GGML_వాడని(size);
        కొరకు (పూర్ణాంకం i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_బఫరు & b = బఫరు_pool[i];
            ఒకవేళ (b.ptr != ptr) {
                కొనసాగు;
            }
            b.వాడిన      = అబద్ధం;
            b.చివరి_వాడిన = std::chrono::steady_clock::now();
#నిర్వచితమైతే DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_కొలత, 1048576) / 1048576));
#అయితే_ముగింపు
            తిరిగి;
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
నిర్మాణం ggml_cann_pool_vmm : బహిరంగ ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    size_t max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    పూర్ణాంకం device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    శూన్య * pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_వాడిన = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_కొలత = 0;

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
    std::vector<శూన్య *> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    స్పష్టమైన ggml_cann_pool_vmm(పూర్ణాంకం device) : device(device) {
        స్వయంచాలక dev    = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size    = dev.మొత్తం_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        ఒకవేళ (pool_addr != 0) {
            కొరకు (స్వయంచాలక & స్థానభ్రంశం : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(స్థానభ్రంశం));
            }
            కొరకు (స్వయంచాలక & handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrt_Release_Mem_చిరునామా(pool_addr));
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
    శూన్య * alloc(size_t size, size_t * actual_కొలత) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        స్థిరరాశి size_t alignment = 128;
        size                   = GGML_PAD(size, alignment);
        ఒకవేళ (size == 0) {
            size = alignment;
        }

        size_t avail = pool_కొలత - pool_వాడిన;

        ఒకవేళ (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_కొలత = size - avail;
            reserve_కొలత        = GGML_PAD(reserve_కొలత, granularity);

            GGML_ASSERT(pool_కొలత + reserve_కొలత <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
            prop.కేటాయింపు_Type       = ACL_MEM_కేటాయింపు_TYPE_PINNED;
            prop.memAttr              = ACL_HBM_MEM_HUGE;
            prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id          = device;
            prop.reserve              = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_కొలత, &prop, 0));

            // reserve virtual address space (if not already reserved)
            ఒకవేళ (pool_addr == 0) {
                ACL_CHECK(aclrt_Reserve_Mem_చిరునామా(&pool_addr, max_size, 0, ఖాళీ, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((అక్షరం *) pool_addr + pool_కొలత, reserve_కొలత, 0, handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((అక్షరం *) pool_addr + pool_కొలత);

            // add to the pool
            pool_కొలత += reserve_కొలత;

#నిర్వచితమైతే DEBUG_CANN_MALLOC
            GGML_LOG_INFO("cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n", device,
                          (గుర్తులేని దీర్ఘ దీర్ఘ) (pool_కొలత / 1024 / 1024),
                          (గుర్తులేని దీర్ఘ దీర్ఘ) (reserve_కొలత / 1024 / 1024));
#అయితే_ముగింపు
        }

        GGML_ASSERT(pool_addr != 0);

        శూన్య * ptr   = (శూన్య *) ((అక్షరం *) pool_addr + pool_వాడిన);
        *actual_కొలత = size;
        pool_వాడిన += size;

#నిర్వచితమైతే DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device, (గుర్తులేని దీర్ఘ దీర్ఘ) size,
                      (గుర్తులేని దీర్ఘ దీర్ఘ) ptr);
#అయితే_ముగింపు
        తిరిగి ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    శూన్య విడుదల(శూన్య * ptr, size_t size) override {
#నిర్వచితమైతే DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device, (గుర్తులేని దీర్ఘ దీర్ఘ) size,
                      (గుర్తులేని దీర్ఘ దీర్ఘ) ptr);
#అయితే_ముగింపు

        pool_వాడిన -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (శూన్య *) ((అక్షరం *) pool_addr + pool_వాడిన));
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
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(పూర్ణాంకం device) {
    std::string mem_pool_type = get_env_as_lowercase("GGML_CANN_MEM_POOL").value_or("");

    ఒకవేళ (mem_pool_type == "prio") {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        తిరిగి std::unique_ptr<ggml_cann_pool>(కొత్త ggml_cann_pool_buf_prio(device));
    }

    ఒకవేళ (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        తిరిగి std::unique_ptr<ggml_cann_pool>(కొత్త ggml_cann_pool_vmm(device));
    }

    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    తిరిగి std::unique_ptr<ggml_cann_pool>(కొత్త ggml_cann_pool_buf(device));
}

// cann buffer

/**
 * @brief Tracks multi-threaded write progress for a single tensor.
 *
 * When multiple threads call set_tensor on different chunks of the same tensor,
 * this tracker accumulates progress and defers post-processing (quantized format
 * transform or ND-to-NZ conversion) until all data has been written.
 */
నిర్మాణం Tensor_సమితి_Tracker {
    std::mutex mtx;                   ///< Protects concurrent access to this tracker
    size_t బైట్లు_written = 0;         ///< Accumulated bytes written so far
    size_t మొత్తం_బైట్లు = 0;           ///< Target size (full tensor)
    std::vector<uint8_t> host_బఫరు; ///< Host staging buffer for quantized tensors
};

/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
నిర్మాణం ggml_backend_cann_బఫరు_context {
    int32_t device;             ///< The device ID associated with this buffer context.
    శూన్య *  dev_ptr = శూన్యసూచిక;  ///< Pointer to the device memory allocated for the buffer.

    std::mutex tracker_mutex;   ///< Protects the trackers map
    std::unordered_map<శూన్య *, std::unique_ptr<Tensor_సమితి_Tracker>> trackers;

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_బఫరు_context(int32_t device, శూన్య * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_బఫరు_context() { ACL_CHECK(aclrtFree(dev_ptr)); }

    /**
     * @brief Get or create a tracker for the given tensor.
     */
    Tensor_సమితి_Tracker * get_or_create_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> తాళం(tracker_mutex);
        స్వయంచాలక కీలకం = tensor->దత్తాంశం;
        స్వయంచాలక it = trackers.find(కీలకం);
        ఒకవేళ (it == trackers.end()) {
            స్వయంచాలక tracker = std::make_unique<Tensor_సమితి_Tracker>();
            tracker->మొత్తం_బైట్లు = ggml_nbytes(tensor);
            స్వయంచాలక * ptr = tracker.get();
            trackers[కీలకం] = std::move(tracker);
            తిరిగి ptr;
        }
        తిరిగి it->second.get();
    }

    /**
     * @brief Remove the tracker for the given tensor.
     */
    శూన్య remove_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> తాళం(tracker_mutex);
        trackers.erase(tensor->దత్తాంశం);
    }
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
నిర్మాణం ggml_backend_cann_బఫరు_type_context {
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
స్థిర స్థిరరాశి అక్షరం * ggml_backend_cann_బఫరు_type_name(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_cann_బఫరు_type_context * buft_ctx = (ggml_backend_cann_బఫరు_type_context *) buft->context;

    తిరిగి buft_ctx->name.c_str();
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
స్థిర తార్కిక ggml_backend_buft_is_cann(ggml_backend_బఫరు_type_t buft) {
    తిరిగి buft->iface.get_name == ggml_backend_cann_బఫరు_type_name;
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
స్థిర శూన్య ggml_backend_cann_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_cann_బఫరు_context * ctx = (ggml_backend_cann_బఫరు_context *) బఫరు->context;
    తొలగించు ctx;
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
స్థిర శూన్య * ggml_backend_cann_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_cann_బఫరు_context * ctx = (ggml_backend_cann_బఫరు_context *) బఫరు->context;
    తిరిగి ctx->dev_ptr;
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
స్థిర శూన్య ggml_backend_cann_transform_q4_0(ggml_tensor * tensor, స్థిరరాశి శూన్య * src, శూన్య * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_బైట్లు = n_elems * పరిమాణం(uint8_t) / 2;

    uint8_t *  quant_స్థానభ్రంశం = (uint8_t *) dst;
    uint16_t * scale_స్థానభ్రంశం = (uint16_t *) ((అక్షరం *) dst + quant_బైట్లు);

    కొరకు (పూర్ణాంకం i = 0; i < groups; i++) {
        స్థిరరాశి ఖండం_q4_0 * సమూహం = (స్థిరరాశి ఖండం_q4_0 *) ((స్థిరరాశి అక్షరం *) src + i * పరిమాణం(ఖండం_q4_0));
        *scale_స్థానభ్రంశం            = సమూహం->d;
        scale_స్థానభ్రంశం++;

        // 0-15
        కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_స్థానభ్రంశం) = (సమూహం->qs[j] & 0x0F);
            (*quant_స్థానభ్రంశం) |= ((సమూహం->qs[j + 1] << 4));
            quant_స్థానభ్రంశం++;
        }

        // 16-31
        కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_స్థానభ్రంశం) = (సమూహం->qs[j] >> 4);
            (*quant_స్థానభ్రంశం) |= (సమూహం->qs[j + 1] & 0xF0);
            quant_స్థానభ్రంశం++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    కొరకు (quant_స్థానభ్రంశం = (uint8_t *) dst; quant_స్థానభ్రంశం < (uint8_t *) dst + quant_బైట్లు; quant_స్థానభ్రంశం++) {
        (*quant_స్థానభ్రంశం) ^= 0x88;
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
స్థిర శూన్య ggml_backend_cann_transform_వెనుక_q4_0(స్థిరరాశి ggml_tensor * tensor, శూన్య * src, శూన్య * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_బైట్లు = n_elems * పరిమాణం(uint8_t) / 2;

    uint8_t *  quant_స్థానభ్రంశం = (uint8_t *) src;
    uint16_t * scale_స్థానభ్రంశం = (uint16_t *) ((అక్షరం *) src + quant_బైట్లు);

    కొరకు (; quant_స్థానభ్రంశం < (uint8_t *) src + quant_బైట్లు; quant_స్థానభ్రంశం++) {
        (*quant_స్థానభ్రంశం) ^= 0x88;
    }
    quant_స్థానభ్రంశం = (uint8_t *) src;

    కొరకు (పూర్ణాంకం i = 0; i < groups; i++) {
        ఖండం_q4_0 * సమూహం = (ఖండం_q4_0 *) ((అక్షరం *) dst + i * పరిమాణం(ఖండం_q4_0));
        సమూహం->d           = *scale_స్థానభ్రంశం;
        scale_స్థానభ్రంశం++;

        // 0-15
        కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 2; j += 2) {
            సమూహం->qs[j]     = ((*quant_స్థానభ్రంశం) & 0x0F);
            సమూహం->qs[j + 1] = ((*quant_స్థానభ్రంశం) >> 4);
            quant_స్థానభ్రంశం++;
        }

        // 16-31
        కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 2; j += 2) {
            సమూహం->qs[j] |= ((*quant_స్థానభ్రంశం) << 4);
            సమూహం->qs[j + 1] |= ((*quant_స్థానభ్రంశం) & 0xF0);
            quant_స్థానభ్రంశం++;
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
స్థిర శూన్య ggml_backend_cann_transform_q8_0(ggml_tensor * tensor, స్థిరరాశి శూన్య * src, శూన్య * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_బైట్లు = n_elems * పరిమాణం(uint8_t);

    uint8_t *  quant_స్థానభ్రంశం = (uint8_t *) dst;
    uint16_t * scale_స్థానభ్రంశం = (uint16_t *) ((అక్షరం *) dst + quant_బైట్లు);

    కొరకు (పూర్ణాంకం i = 0; i < groups; i++) {
        స్థిరరాశి ఖండం_q8_0 * సమూహం = (స్థిరరాశి ఖండం_q8_0 *) ((స్థిరరాశి అక్షరం *) src + i * పరిమాణం(ఖండం_q8_0));
        *scale_స్థానభ్రంశం            = సమూహం->d;
        scale_స్థానభ్రంశం++;
        size_t సమూహం_quant_కొలత = QK8_0 * పరిమాణం(uint8_t);
        జ్ఞాపకం_నకలు(quant_స్థానభ్రంశం, సమూహం->qs, సమూహం_quant_కొలత);
        quant_స్థానభ్రంశం += సమూహం_quant_కొలత;
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
స్థిర శూన్య ggml_backend_cann_transform_వెనుక_q8_0(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి శూన్య * src, శూన్య * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_బైట్లు = n_elems * పరిమాణం(uint8_t);

    స్థిరరాశి uint8_t *  quant_స్థానభ్రంశం = (స్థిరరాశి uint8_t *) src;
    స్థిరరాశి uint16_t * scale_స్థానభ్రంశం = (స్థిరరాశి uint16_t *) ((స్థిరరాశి అక్షరం *) src + quant_బైట్లు);

    కొరకు (పూర్ణాంకం i = 0; i < groups; i++) {
        ఖండం_q8_0 * సమూహం = (ఖండం_q8_0 *) ((అక్షరం *) dst + i * పరిమాణం(ఖండం_q8_0));
        సమూహం->d           = *scale_స్థానభ్రంశం;
        scale_స్థానభ్రంశం++;
        size_t సమూహం_quant_కొలత = QK8_0 * పరిమాణం(uint8_t);
        జ్ఞాపకం_నకలు(సమూహం->qs, quant_స్థానభ్రంశం, సమూహం_quant_కొలత);
        quant_స్థానభ్రంశం += సమూహం_quant_కొలత;
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
స్థిర శూన్య ggml_backend_cann_transform(ggml_tensor * tensor, స్థిరరాశి శూన్య * src, శూన్య * dst) {
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            ఆపు;
        అప్రమేయ:
            ఆపు;
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
స్థిర శూన్య ggml_backend_cann_transform_వెనుక(స్థిరరాశి ggml_tensor * tensor, శూన్య * src, శూన్య * dst) {
    ఎంపిక (tensor->type) {
        స్థితి GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_వెనుక_q4_0(tensor, src, dst);
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_వెనుక_q8_0(tensor, src, dst);
            ఆపు;
        అప్రమేయ:
            ఆపు;
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
స్థిర తార్కిక need_transform(ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
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
స్థిర గణన ggml_స్థితి ggml_backend_cann_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor) {
    ఒకవేళ (tensor->view_src != ఖాళీ && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->బఫరు->buft == బఫరు->buft);
        తిరిగి GGML_స్థితి_SUCCESS;
    }

    // TODO: cann backend doesn't support quantized yet. Just leave the code
    // here.
    ఒకవేళ (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_కొలత = ggml_nbytes(tensor);
        size_t padded_కొలత   = ggml_backend_buft_get_alloc_కొలత(బఫరు->buft, tensor);

        ఒకవేళ (padded_కొలత > original_కొలత && tensor->view_src == శూన్యసూచిక) {
            size_t memset_కొలత = padded_కొలత - original_కొలత;
            ACL_CHECK(aclrtMemset((అక్షరం *) tensor->దత్తాంశం + original_కొలత, memset_కొలత, 0, memset_కొలత));
        }
    }
    తిరిగి GGML_స్థితి_SUCCESS;
}

/**
 * @brief Workspace for caching NZ buffers per device.
 *
 * This struct manages a device buffer used in NZ computations. It supports
 * allocation, reallocation, and clearing of cached memory. The struct is
 * designed to be used with a global array, one per device.
 */
నిర్మాణం ggml_cann_nz_workspace {
    std::mutex mtx;    // Protects ptr/allocated from concurrent access
    శూన్య * ptr;        // Pointer to allocated device buffer
    size_t allocated;  // Size of currently allocated buffer in bytes

    /**
     * @brief Constructor. Initializes the workspace with no allocated memory.
     */
    ggml_cann_nz_workspace() : ptr(శూన్యసూచిక), allocated(0) {}

    /**
     * @brief Free cached memory and reset the workspace.
     *
     * If a buffer has been allocated, this function releases it using
     * aclrtFree and resets internal state.
     */
    శూన్య clear() {
        ఒకవేళ (ptr) {
            ACL_CHECK(aclrtFree(ptr));
            ptr       = శూన్యసూచిక;
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
    శూన్య realloc(size_t new_కొలత) {
        ఒకవేళ (new_కొలత > allocated) {
            clear();
            ACL_CHECK(aclrtMalloc(&ptr, new_కొలత, ACL_MEM_MALLOC_HUGE_మొదటి));
            allocated = new_కొలత;
        }
    }

    /**
     * @brief Get the device buffer pointer.
     *
     * @return Pointer to the allocated buffer, or nullptr if not allocated.
     */
    శూన్య * get() స్థిరరాశి { తిరిగి ptr; }
};

/**
 * @brief Global array of NZ workspaces, one per device.
 */
స్థిర ggml_cann_nz_workspace g_nz_workspaces[GGML_CANN_MAX_DEVICES];

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
స్థిర శూన్య బరువు_format_గమ్యం_nz(ggml_tensor * tensor, పూర్ణాంకం device) {
    acl_tensor_ptr బరువు_Transposed = ggml_cann_create_tensor(tensor, tensor->ne, tensor->nb, 2, ACL_FORMAT_ND, 0);
    uint64_t       workspace_కొలత    = 0;
    aclOpExecutor * executor;

    // TransMatmulWeight
    ACL_CHECK(aclnn_Trans_Matmul_బరువు_Get_Workspace_కొలత(బరువు_Transposed.get(), &workspace_కొలత, &executor));

    std::lock_guard<std::mutex> తాళం(g_nz_workspaces[device].mtx);
    // Avoid frequent malloc/free of the workspace.
    g_nz_workspaces[device].realloc(workspace_కొలత);

    శూన్య * g_nz_workspace = g_nz_workspaces[device].get();

    ACL_CHECK(aclnn_Trans_Matmul_బరువు(g_nz_workspace, workspace_కొలత, executor, శూన్యసూచిక));
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
స్థిర శూన్య ggml_backend_cann_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                                ggml_tensor *         tensor,
                                                స్థిరరాశి శూన్య *          దత్తాంశం,
                                                size_t                స్థానభ్రంశం,
                                                size_t                size) {
    ggml_backend_cann_బఫరు_context * ctx = (ggml_backend_cann_బఫరు_context *) బఫరు->context;

    ggml_cann_సమితి_device(ctx->device);

    // Only check env once.
    స్థిర తార్కిక బరువు_గమ్యం_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    తార్కిక is_quantized = need_transform(tensor->type);
    తార్కిక is_nz        = !is_quantized && tensor->type != GGML_TYPE_BF16 && బరువు_గమ్యం_nz &&
                 is_matmul_బరువు((స్థిరరాశి ggml_tensor *) tensor);

    // Plain tensor (not quantized, not NZ): direct copy, no tracking needed
    ఒకవేళ (!is_quantized && !is_nz) {
        ACL_CHECK(aclrtMemcpy((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size, దత్తాంశం, size, ACL_MEMCPY_HOST_గమ్యం_DEVICE));
        తిరిగి;
    }

    // Single-shot write (full tensor at once): handle directly without tracking overhead
    ఒకవేళ (స్థానభ్రంశం == 0 && size == ggml_nbytes(tensor)) {
        ఒకవేళ (is_quantized) {
            శూన్య * transform_బఫరు = జ్ఞాపకం_ఇవ్వు(size);
            ggml_backend_cann_transform(tensor, దత్తాంశం, transform_బఫరు);
            ACL_CHECK(aclrtMemcpy(tensor->దత్తాంశం, size, transform_బఫరు, size, ACL_MEMCPY_HOST_గమ్యం_DEVICE));
            విడుదల(transform_బఫరు);
        } లేకపోతే {
            // NZ weight
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ACL_CHECK(aclrtMemcpy(tensor->దత్తాంశం, size, దత్తాంశం, size, ACL_MEMCPY_HOST_గమ్యం_DEVICE));
            బరువు_format_గమ్యం_nz(tensor, ctx->device);
        }
        తిరిగి;
    }

    // Chunked write: use tracker to accumulate progress and defer transform/conversion
    Tensor_సమితి_Tracker * tracker = ctx->get_or_create_tracker(tensor);
    std::unique_lock<std::mutex> తాళం(tracker->mtx);

    ఒకవేళ (is_quantized) {
        // Stage data in host buffer; transform requires full tensor data
        ఒకవేళ (tracker->host_బఫరు.ఖాళీలు()) {
            tracker->host_బఫరు.resize(tracker->మొత్తం_బైట్లు);
        }
        జ్ఞాపకం_నకలు(tracker->host_బఫరు.దత్తాంశం() + స్థానభ్రంశం, దత్తాంశం, size);
    } లేకపోతే {
        // NZ weight: upload chunk to device immediately, defer conversion
        ACL_CHECK(aclrtMemcpy((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size, దత్తాంశం, size, ACL_MEMCPY_HOST_గమ్యం_DEVICE));
    }

    tracker->బైట్లు_written += size;

    // All chunks received: perform deferred transform/conversion
    ఒకవేళ (tracker->బైట్లు_written >= tracker->మొత్తం_బైట్లు) {
        ఒకవేళ (is_quantized) {
            శూన్య * transform_బఫరు = జ్ఞాపకం_ఇవ్వు(tracker->మొత్తం_బైట్లు);
            ggml_backend_cann_transform(tensor, tracker->host_బఫరు.దత్తాంశం(), transform_బఫరు);
            ACL_CHECK(aclrtMemcpy(tensor->దత్తాంశం, tracker->మొత్తం_బైట్లు, transform_బఫరు, tracker->మొత్తం_బైట్లు, ACL_MEMCPY_HOST_గమ్యం_DEVICE));
            విడుదల(transform_బఫరు);
        }

        ఒకవేళ (is_nz) {
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            బరువు_format_గమ్యం_nz(tensor, ctx->device);
        }

        // Unlock before removing tracker, as remove_tracker destroys the mutex
        తాళం.unlock();
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
స్థిర శూన్య ggml_backend_cann_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు,
                                                స్థిరరాశి ggml_tensor *   tensor,
                                                శూన్య *                దత్తాంశం,
                                                size_t                స్థానభ్రంశం,
                                                size_t                size) {
    ggml_backend_cann_బఫరు_context * ctx = (ggml_backend_cann_బఫరు_context *) బఫరు->context;

    ggml_cann_సమితి_device(ctx->device);

    ఒకవేళ (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(దత్తాంశం, size, (అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size, ACL_MEMCPY_DEVICE_గమ్యం_HOST));
    } లేకపోతే {
        శూన్య * transform_బఫరు = జ్ఞాపకం_ఇవ్వు(size);
        ACL_CHECK(aclrtMemcpy(transform_బఫరు, size, (అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size, ACL_MEMCPY_DEVICE_గమ్యం_HOST));
        ggml_backend_cann_transform_వెనుక(tensor, transform_బఫరు, దత్తాంశం);
        విడుదల(transform_బఫరు);
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
స్థిర తార్కిక ggml_backend_cann_బఫరు_cpy_tensor(ggml_backend_బఫరు_t బఫరు,
                                                స్థిరరాశి ggml_tensor *   src,
                                                ggml_tensor *         dst) {
    ఒకవేళ (ggml_backend_buft_is_cann(src->బఫరు->buft)) {
        ggml_backend_cann_బఫరు_context * src_ctx = (ggml_backend_cann_బఫరు_context *) src->బఫరు->context;
        ggml_backend_cann_బఫరు_context * dst_ctx = (ggml_backend_cann_బఫరు_context *) బఫరు->context;

        size_t memcpy_కొలత = ggml_nbytes(src);
        // Same device.
        ఒకవేళ (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((అక్షరం *) dst->దత్తాంశం, memcpy_కొలత, (స్థిరరాశి అక్షరం *) src->దత్తాంశం, memcpy_కొలత,
                                  ACL_MEMCPY_DEVICE_గమ్యం_DEVICE));
            తిరిగి నిజం;
        } లేకపోతే {
#నిర్వచితమైతే ASCEND_310P
            // TODO: Support 310p P2P copy
            తిరిగి అబద్ధం;
#అయితే_ముగింపు
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device, dst_ctx->device));
            ఒకవేళ (canAccessPeer) {
                ggml_cann_సమితి_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((అక్షరం *) dst->దత్తాంశం, memcpy_కొలత, (స్థిరరాశి అక్షరం *) src->దత్తాంశం, memcpy_కొలత,
                                      ACL_MEMCPY_DEVICE_గమ్యం_DEVICE));
                తిరిగి నిజం;
            }
        }
    }
    తిరిగి అబద్ధం;
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
స్థిర శూన్య ggml_backend_cann_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    ggml_backend_cann_బఫరు_context * ctx = (ggml_backend_cann_బఫరు_context *) బఫరు->context;

    ggml_cann_సమితి_device(ctx->device);
    ACL_CHECK(aclrtMemset((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size, value, size));
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
స్థిర శూన్య ggml_backend_cann_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    ggml_backend_cann_బఫరు_context * ctx = (ggml_backend_cann_బఫరు_context *) బఫరు->context;

    ggml_cann_సమితి_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, బఫరు->size, value, బఫరు->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
స్థిర స్థిరరాశి ggml_backend_బఫరు_i ggml_backend_cann_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_cann_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_cann_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_cann_బఫరు_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cann_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cann_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_cann_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ggml_backend_cann_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
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
స్థిర ggml_backend_బఫరు_t ggml_backend_cann_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    ggml_backend_cann_బఫరు_type_context * buft_ctx = (ggml_backend_cann_బఫరు_type_context *) buft->context;

    ggml_cann_సమితి_device(buft_ctx->device);

    స్థిరరాశి size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    ఒకవేళ (size == 0) {
        size = alignment;
    }
    శూన్య *   dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_మొదటి);
    ఒకవేళ (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, buft_ctx->device, acl_Get_Recent_Err_సందేశం());
        తిరిగి శూన్యసూచిక;
    }

    ggml_backend_cann_బఫరు_context * ctx = కొత్త ggml_backend_cann_బఫరు_context(buft_ctx->device, dev_ptr);

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_cann_బఫరు_interface, ctx, size);
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
స్థిర size_t ggml_backend_cann_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి 128;

    GGML_వాడని(buft);
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
స్థిర size_t ggml_backend_cann_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft,
                                                           స్థిరరాశి ggml_tensor *        tensor) {
    size_t  size = ggml_nbytes(tensor);
    int64_t ne0  = tensor->ne[0];

    // Only check env once.
    స్థిర తార్కిక బరువు_గమ్యం_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);
    ఒకవేళ (ggml_is_quantized(tensor->type)) {
        ఒకవేళ (ne0 % మాతృక_అడ్డువరుస_PADDING != 0) {
            size += ggml_అడ్డువరుస_కొలత(tensor->type, మాతృక_అడ్డువరుస_PADDING - ne0 % మాతృక_అడ్డువరుస_PADDING);
        }
    } లేకపోతే ఒకవేళ (బరువు_గమ్యం_nz && tensor->type != GGML_TYPE_BF16
               && is_matmul_బరువు((స్థిరరాశి ggml_tensor *) tensor)) {
        // NZ format weight are not support quantized yet.
        // If ND tensor transform to NZ, size may changed.
        int64_t shape[] = { tensor->ne[1], tensor->ne[0] };
        GGML_ASSERT(tensor->ne[2] == 1);
        GGML_ASSERT(tensor->ne[3] == 1);
        స్థిరరాశి aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
        size_t              new_కొలత;
        ACL_CHECK(aclnn_Calculate_Matmul_బరువు_కొలత_V2(acl_shape, ggml_cann_type_mapping(tensor->type), &new_కొలత));
        ACL_CHECK(aclDestroyIntArray(acl_shape));
        size = std::max(size, new_కొలత);
    }

    తిరిగి size;

    GGML_వాడని(buft);
}

స్థిర తార్కిక ggml_backend_cann_బఫరు_type_is_host(ggml_backend_బఫరు_type_t buft) {
    తిరిగి అబద్ధం;

    GGML_వాడని(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
స్థిర స్థిరరాశి ggml_backend_బఫరు_type_i ggml_backend_cann_బఫరు_type_interface = {
    /* .get_name            = */ ggml_backend_cann_బఫరు_type_name,
    /* .alloc_buffer        = */ ggml_backend_cann_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ఖాళీ,
    /* .get_alignment       = */ ggml_backend_cann_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ఖాళీ,  // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cann_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ggml_backend_cann_బఫరు_type_is_host,
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
ggml_backend_బఫరు_type_t ggml_backend_cann_బఫరు_type(int32_t device) {
    స్థిర std::mutex           mutex;
    std::lock_guard<std::mutex> తాళం(mutex);

    ఒకవేళ (device >= ggml_backend_cann_get_device_లెక్క()) {
        తిరిగి శూన్యసూచిక;
    }

    స్థిర ggml_backend_బఫరు_type ggml_backend_cann_బఫరు_types[GGML_CANN_MAX_DEVICES];

    స్థిర తార్కిక ggml_backend_cann_బఫరు_type_initialized = అబద్ధం;

    ఒకవేళ (!ggml_backend_cann_బఫరు_type_initialized) {
        కొరకు (int32_t i = 0; i < ggml_cann_info().device_లెక్క; i++) {
            ggml_backend_cann_బఫరు_types[i] = {
                /* .iface    = */ ggml_backend_cann_బఫరు_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), i),
                /* .context  = */
                కొత్త ggml_backend_cann_బఫరు_type_context{ i, "CANN" + std::to_string(i) },
            };
        }
        ggml_backend_cann_బఫరు_type_initialized = నిజం;
    }

    తిరిగి &ggml_backend_cann_బఫరు_types[device];
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
స్థిర స్థిరరాశి అక్షరం * ggml_backend_cann_host_బఫరు_type_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి "CANN_Host";

    GGML_వాడని(buft);
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
స్థిర స్థిరరాశి అక్షరం * ggml_backend_cann_host_బఫరు_name(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి "CANN_Host";

    GGML_వాడని(బఫరు);
}

/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer, including
 * its context.
 *
 * @param buffer The CANN host buffer to free.
 */
స్థిర శూన్య ggml_backend_cann_host_బఫరు_free(ggml_backend_బఫరు_t బఫరు) {
    ACL_CHECK(aclrtFreeHost(బఫరు->context));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
స్థిర శూన్య * ggml_cann_host_malloc(size_t size) {
    ఒకవేళ (getenv("GGML_CANN_NO_PINNED") != శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }

    స్థిరరాశి size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    ఒకవేళ (size == 0) {
        size = alignment;
    }

    శూన్య *   hostPtr = శూన్యసూచిక;
    aclError err     = aclrtMallocHost((శూన్య **) &hostPtr, size);
    ఒకవేళ (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                      acl_Get_Recent_Err_సందేశం());
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
స్థిర ggml_backend_బఫరు_t ggml_backend_cann_host_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft,
                                                                             size_t                     size) {
    శూన్య * hostPtr = ggml_cann_host_malloc(size);

    ఒకవేళ (hostPtr == శూన్యసూచిక) {
        // fallback to cpu buffer
        తిరిగి ggml_backend_buft_alloc_బఫరు(ggml_backend_cpu_బఫరు_type(), size);
    }

    ggml_backend_బఫరు_t బఫరు = ggml_backend_cpu_బఫరు_from_ptr(hostPtr, size);
    బఫరు->buft                 = buft;
    బఫరు->iface.free_బఫరు    = ggml_backend_cann_host_బఫరు_free;

    తిరిగి బఫరు;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_బఫరు_type_t ggml_backend_cann_host_బఫరు_type() {
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_cann_బఫరు_type_host = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cann_host_బఫరు_type_name,
                           /* .alloc_buffer         = */ ggml_backend_cann_host_బఫరు_type_alloc_బఫరు,
                           /* .alloc_buffer_n       = */ ఖాళీ,
                           /* .get_alignment        = */ ggml_backend_cpu_బఫరు_type()->iface.get_alignment,
                           /* .get_max_size         = */ ఖాళీ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_బఫరు_type()->iface.get_alloc_కొలత,
                           /* .get_alloc_size_n     = */ ఖాళీ,
                           /* .is_host              = */ ggml_backend_cpu_బఫరు_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
        /* .context  = */ శూన్యసూచిక,
    };

    తిరిగి &ggml_backend_cann_బఫరు_type_host;
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
స్థిర తార్కిక ggml_cann_compute_forward(ggml_backend_cann_context & ctx, నిర్మాణం ggml_tensor * dst) {
    ఎంపిక (dst->op) {
        స్థితి GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            ggml_cann_get_అడ్డువరుసలు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            ggml_cann_సమితి_అడ్డువరుసలు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_ADD1:
            ggml_cann_ద్విమాన_op<aclnn_కూడు>(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SUB:
            ggml_cann_ద్విమాన_op<aclnn_sub>(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_MUL:
            ggml_cann_ద్విమాన_op<aclnn_mul>(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DIV:
            ggml_cann_ద్విమాన_op<aclnn_div>(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(dst)) {
                స్థితి GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_OP_UNARY(Abs);
                    ఆపు;
                స్థితి GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_OP_UNARY(Neg);
                    ఆపు;
                స్థితి GGML_UNARY_OP_GELU:
                స్థితి GGML_UNARY_OP_GELU_ERF:
                    // aclnnGelu internally uses the erf-based approximation.
                    GGML_CANN_CALL_OP_UNARY(Gelu);
                    ఆపు;
                స్థితి GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_OP_UNARY(Silu);
                    ఆపు;
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                    {
                        స్వయంచాలక lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                        };
                        ggml_cann_op_unary(lambda, ctx, dst);
                    }
                    ఆపు;
                స్థితి GGML_UNARY_OP_TANH:
                    GGML_CANN_CALL_OP_UNARY(Tanh);
                    ఆపు;
                స్థితి GGML_UNARY_OP_RELU:
                    GGML_CANN_CALL_OP_UNARY(Relu);
                    ఆపు;
                స్థితి GGML_UNARY_OP_SIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Sigmoid);
                    ఆపు;
                స్థితి GGML_UNARY_OP_HARDSIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Hardsigmoid);
                    ఆపు;
                స్థితి GGML_UNARY_OP_HARDSWISH:
                    GGML_CANN_CALL_OP_UNARY(Hardswish);
                    ఆపు;
                స్థితి GGML_UNARY_OP_EXP:
                    GGML_CANN_CALL_OP_UNARY(Exp);
                    ఆపు;
                స్థితి GGML_UNARY_OP_ELU:
                    ggml_cann_elu(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_SGN:
                    GGML_CANN_CALL_OP_UNARY(Sign);
                    ఆపు;
                స్థితి GGML_UNARY_OP_అడుగు:
                    ggml_cann_అడుగు(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_SOFTPLUS:
                    ggml_cann_softplus(ctx, dst);
                    ఆపు;
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(dst)) {
                స్థితి GGML_GLU_OP_REGLU:
                    GGML_CANN_CALL_OP_UNARY_GATED(Relu);
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU:
                    ggml_cann_geglu(ctx, dst, 0);  // approximate=0 → tanh
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                    ggml_cann_geglu(ctx, dst, 1);  // approximate=1 → erf
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU:
                    ggml_cann_swiglu(ctx, dst);
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cann_swiglu_clamp(ctx, dst);
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                    ggml_cann_geglu_శీఘ్ర(ctx, dst);
                    ఆపు;
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_సమూహం_NORM:
            ggml_cann_సమూహం_norm(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_L2_NORM:
            ggml_cann_l2_norm(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cann_cross_entropy_loss(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == శూన్యసూచిక);
            dst->src[1] = dst->src[0];
            ggml_cann_ద్విమాన_op<aclnn_mul>(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SQRT:
            GGML_CANN_CALL_OP_UNARY(Sqrt);
            ఆపు;
        స్థితి GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_సమితి:
            ggml_cann_సమితి(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
            ఆపు;
        స్థితి GGML_OP_DIAG_ముసుగు_INF:
            ggml_cann_diag_ముసుగు(ctx, dst, -INFINITY);
            ఆపు;
        స్థితి GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_IM2_నిలువు:
            ggml_cann_im2col(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_కూడిక:
            ggml_cann_కూడిక(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            ggml_cann_కూడిక_అడ్డువరుసలు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ARGMAX:
            ggml_cann_argmax(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_COS:
            ggml_cann_op_unary<aclnn_cos>(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SIN:
            ggml_cann_op_unary<aclnn_sin>(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cann_conv_transpose_1d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_LOG:
            GGML_CANN_CALL_OP_UNARY(Log);
            ఆపు;
        స్థితి GGML_OP_సగటు:
            ggml_cann_సగటు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_PAD_REFLECT_1D:
            ggml_cann_pad_reflect_1d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_లెక్క_EQUAL:
            ggml_cann_లెక్క_equal(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_బహిర్గతం_PROD:
            ggml_cann_బహిర్గతం_prod(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
            ggml_cann_gated_రేఖీయ_attn(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SSM_CONV:
            ggml_cann_ssm_conv(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CUMSUM:
            ggml_cann_cumsum(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_TRI:
            ggml_cann_tri(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_FILL:
            ggml_cann_fill(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DIAG:
            ggml_cann_diag(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SOLVE_TRI:
            ggml_cann_solve_tri(ctx, dst);
            ఆపు;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
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
స్థిర స్థిరరాశి అక్షరం * ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;

    తిరిగి cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
స్థిర శూన్య ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    తొలగించు cann_ctx;
    తొలగించు backend;
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
స్థిర శూన్య ggml_backend_cann_సమితి_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *  tensor,
                                               స్థిరరాశి శూన్య *   దత్తాంశం,
                                               size_t         స్థానభ్రంశం,
                                               size_t         size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_బఫరు_t       buf      = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;

    GGML_ASSERT(buf->buft == ggml_backend_cann_బఫరు_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size, దత్తాంశం, size, ACL_MEMCPY_HOST_గమ్యం_DEVICE,
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
స్థిర శూన్య ggml_backend_cann_get_tensor_async(ggml_backend_t      backend,
                                               స్థిరరాశి ggml_tensor * tensor,
                                               శూన్య *              దత్తాంశం,
                                               size_t              స్థానభ్రంశం,
                                               size_t              size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_బఫరు_t       buf      = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;

    GGML_ASSERT(buf->buft == ggml_backend_cann_బఫరు_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync(దత్తాంశం, size, (అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size, ACL_MEMCPY_DEVICE_గమ్యం_HOST,
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
స్థిర తార్కిక ggml_backend_cann_cpy_tensor_async(ggml_backend_t      backend_src,
                                               ggml_backend_t      backend_dst,
                                               స్థిరరాశి ggml_tensor * src,
                                               ggml_tensor *       dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) || ggml_backend_is_cann(backend_dst));

    GGML_ASSERT(!is_matmul_బరువు((స్థిరరాశి ggml_tensor *) src));

    ఒకవేళ (!ggml_backend_buft_is_cann(src->బఫరు->buft) || !ggml_backend_buft_is_cann(dst->బఫరు->buft)) {
        తిరిగి అబద్ధం;
    }

    ggml_backend_బఫరు_t buf_src = src->view_src ? src->view_src->బఫరు : src->బఫరు;
    ggml_backend_బఫరు_t buf_dst = dst->view_src ? dst->view_src->బఫరు : dst->బఫరు;

    ggml_backend_cann_context * cann_ctx_src = (ggml_backend_cann_context *) backend_src->context;
    ggml_backend_cann_context * cann_ctx_dst = (ggml_backend_cann_context *) backend_dst->context;

    size_t copy_కొలత = ggml_nbytes(dst);
    ఒకవేళ (copy_కొలత == 0) {
        తిరిగి నిజం;
    }
    ఒకవేళ (backend_src != backend_dst) {
#నిర్వచితమైతే ASCEND_310P
        // TODO: Support 310p P2P copy
        తిరిగి అబద్ధం;
#అయితే_ముగింపు
        ggml_backend_cann_బఫరు_context * buf_ctx_src = (ggml_backend_cann_బఫరు_context *) buf_src->context;
        ggml_backend_cann_బఫరు_context * buf_ctx_dst = (ggml_backend_cann_బఫరు_context *) buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device, cann_ctx_dst->device));
        ఒకవేళ (!canAccessPeer) {
            తిరిగి అబద్ధం;
        }

        // need open both directions for memcpyasync between devices.
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_సమితి_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        // wait for task_queue empty to keep task order.
        ACL_CHECK(aclrtMemcpyAsync(dst->దత్తాంశం, copy_కొలత, src->దత్తాంశం, copy_కొలత, ACL_MEMCPY_DEVICE_గమ్యం_DEVICE,
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
        ACL_CHECK(aclrt_Synchronize_ప్రవాహం(cann_ctx_src->stream()));
    } లేకపోతే {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->దత్తాంశం, copy_కొలత, src->దత్తాంశం, copy_కొలత, ACL_MEMCPY_DEVICE_గమ్యం_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    తిరిగి నిజం;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
స్థిర శూన్య ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_సమితి_device(cann_ctx->device);
    ACL_CHECK(aclrt_Synchronize_ప్రవాహం(cann_ctx->stream()));
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
స్థిర తార్కిక ggml_cann_can_fuse(స్థిరరాశి నిర్మాణం ggml_cgraph *          cgraph,
                               పూర్ణాంకం                                 కణుపు_idx,
                               std::initializer_list<గణన ggml_op> ops) {
    ఒకవేళ (!ggml_can_fuse(cgraph, కణుపు_idx, ops)) {
        తిరిగి అబద్ధం;
    }

    // CANN backend supports fusing ADD + RMS_NORM operations
    ఒకవేళ ((ops.size() == 2) && ops.begin()[0] == GGML_OP_కూడు && ops.begin()[1] == GGML_OP_RMS_NORM) {
        ggml_tensor * కూడు_కణుపు = cgraph->nodes[కణుపు_idx];
        // TODO: support broadcast for ADD + RMS_NORM
        ఒకవేళ (కూడు_కణుపు->src[0]->ne[0] != కూడు_కణుపు->src[1]->ne[0] || కూడు_కణుపు->src[0]->ne[1] != కూడు_కణుపు->src[1]->ne[1] ||
            కూడు_కణుపు->src[0]->ne[2] != కూడు_కణుపు->src[1]->ne[2] || కూడు_కణుపు->src[0]->ne[3] != కూడు_కణుపు->src[1]->ne[3]) {
            తిరిగి అబద్ధం;
        }
        తిరిగి నిజం;
    }

    తిరిగి అబద్ధం;
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
స్థిర శూన్య evaluate_and_capture_cann_లేఖాచిత్రం(ggml_backend_cann_context * cann_ctx,
                                            ggml_cgraph *               cgraph,
                                            తార్కిక                        use_cann_లేఖాచిత్రం,
                                            తార్కిక                        cann_లేఖాచిత్రం_capture_required) {
#నిర్వచితమైతే USE_ACL_లేఖాచిత్రం
    ఒకవేళ (use_cann_లేఖాచిత్రం && cann_లేఖాచిత్రం_capture_required) {  // Begin CANN graph capture
        ACL_CHECK(aclmdlRICaptureBegin(cann_ctx->stream(), ACL_MODEL_RI_CAPTURE_బహుళకం_GLOBAL));
    }
#అయితే_ముగింపు  // USE_ACL_GRAPH
    // Only perform the graph execution if CANN graphs are not enabled, or we are capturing the graph.
    // With the use of CANN graphs, the execution will be performed by the graph launch.
    స్థిర తార్కిక opt_fusion = parse_bool(get_env_as_lowercase("GGML_CANN_OPERATOR_FUSION").value_or(""));

    ఒకవేళ (!use_cann_లేఖాచిత్రం || cann_లేఖాచిత్రం_capture_required) {
        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * కణుపు = cgraph->nodes[i];
            ఒకవేళ (opt_fusion) {
                ఒకవేళ (ggml_cann_can_fuse(cgraph, i, { GGML_OP_కూడు, GGML_OP_RMS_NORM })) {
                    ggml_cann_op_కూడు_rms_norm_fused(*cann_ctx, కణుపు, cgraph->nodes[i + 1]);
                    i++;
                    కొనసాగు;
                }
            }

            ఒకవేళ (ggml_is_ఖాళీలు(కణుపు) || కణుపు->op == GGML_OP_RESHAPE || కణుపు->op == GGML_OP_TRANSPOSE ||
                కణుపు->op == GGML_OP_VIEW || కణుపు->op == GGML_OP_క్రమమార్పు || కణుపు->op == GGML_OP_NONE) {
                కొనసాగు;
            }

            ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                కొనసాగు;
            }

            తార్కిక ok = ggml_cann_compute_forward(*cann_ctx, కణుపు);
            ఒకవేళ (!ok) {
                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, కణుపు->name, ggml_op_name(కణుపు->op));
            }
            GGML_ASSERT(ok);
        }
    }

#నిర్వచితమైతే USE_ACL_లేఖాచిత్రం
    ఒకవేళ (use_cann_లేఖాచిత్రం) {
        GGML_ASSERT(!cann_ctx->లేఖాచిత్రం_lru_cache.cache_list.ఖాళీలు());
        ggml_cann_లేఖాచిత్రం * matched_లేఖాచిత్రం = cann_ctx->లేఖాచిత్రం_lru_cache.cache_list.front();

        ఒకవేళ (cann_లేఖాచిత్రం_capture_required) {  // End CANN graph capture
            ACL_CHECK(aclmdl_RICapture_చివర(cann_ctx->stream(), &matched_లేఖాచిత్రం->graph));
        }

        // Execute CANN graph
        ACL_CHECK(aclmdlRIExecuteAsync(matched_లేఖాచిత్రం->graph, cann_ctx->stream()));
    }
#అయితే_ముగింపు  // USE_ACL_GRAPH
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
స్థిర గణన ggml_స్థితి ggml_backend_cann_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_సమితి_device(cann_ctx->device);
    g_nz_workspaces[cann_ctx->device].clear();

    // calculate rope cache for fist layer in current device.
    cann_ctx->rope_cache.cached = అబద్ధం;

    తార్కిక లేఖాచిత్రం_capture_required = అబద్ధం;
#నిర్వచితమైతే USE_ACL_లేఖాచిత్రం
    తార్కిక use_cann_లేఖాచిత్రం = నిజం;

    స్థిర తార్కిక prefill_use_లేఖాచిత్రం = parse_bool(get_env_as_lowercase("GGML_CANN_PREFILL_USE_GRAPH").value_or(""));
    ఒకవేళ (!prefill_use_లేఖాచిత్రం) {
        // Do not use acl_graph for prefill.
        కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * కణుపు = cgraph->nodes[i];
            // TODO: Optimize here. Currently, we can only
            // get seq_len by FA's input.
            ఒకవేళ (కణుపు->op == GGML_OP_FLASH_ATTN_EXT) {
                // Q -> src[0], shape: [B, S, N, D]
                use_cann_లేఖాచిత్రం = (కణుపు->src[0]->ne[1] == 1);
                ఆపు;
            }
        }
    }

    ఒకవేళ (!cann_ctx->acl_లేఖాచిత్రం_బహుళకం) {
        use_cann_లేఖాచిత్రం = అబద్ధం;
    }

    ఒకవేళ (use_cann_లేఖాచిత్రం) {
        // If no matching graph is found, the graph needs to be recaptured.
        లేఖాచిత్రం_capture_required = !cann_ctx->లేఖాచిత్రం_lru_cache.కనుగొను_and_move_గమ్యం_front(cgraph);

        ఒకవేళ (లేఖాచిత్రం_capture_required) {
            // If no matching graph is found, add a new ACL graph.
            ggml_cann_లేఖాచిత్రం * new_లేఖాచిత్రం = ggml_cann_లేఖాచిత్రం::create_from_cgraph(cgraph);
            cann_ctx->లేఖాచిత్రం_lru_cache.నెట్టు(new_లేఖాచిత్రం);

            // Pre-load rope cache before graph capture.  During capture the
            // stream cannot perform host-to-device memcpy or device memory
            // malloc/free.  Running the full cache init now populates the
            // cache metadata so these branches are skipped during capture,
            // while also warming up the memory pool.
            కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * కణుపు = cgraph->nodes[i];
                ఒకవేళ (కణుపు->op == GGML_OP_ROPE) {
                    ggml_cann_rope_cache_preload(*cann_ctx, కణుపు);
                    ఆపు;
                }
            }
        }
    }
#లేకపోతే
    తార్కిక use_cann_లేఖాచిత్రం = అబద్ధం;
#అయితే_ముగింపు  // USE_ACL_GRAPH
    evaluate_and_capture_cann_లేఖాచిత్రం(cann_ctx, cgraph, use_cann_లేఖాచిత్రం, లేఖాచిత్రం_capture_required);

    తిరిగి GGML_స్థితి_SUCCESS;
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
స్థిర తార్కిక ggml_backend_cann_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ఎంపిక (op->op) {
        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(op)) {
                స్థితి GGML_UNARY_OP_ABS:
                స్థితి GGML_UNARY_OP_NEG:
                స్థితి GGML_UNARY_OP_GELU:
                స్థితి GGML_UNARY_OP_SILU:
                స్థితి GGML_UNARY_OP_RELU:
                స్థితి GGML_UNARY_OP_SIGMOID:
                స్థితి GGML_UNARY_OP_HARDSIGMOID:
                స్థితి GGML_UNARY_OP_HARDSWISH:
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                స్థితి GGML_UNARY_OP_TANH:
                స్థితి GGML_UNARY_OP_EXP:
                స్థితి GGML_UNARY_OP_ELU:
                స్థితి GGML_UNARY_OP_SGN:
                స్థితి GGML_UNARY_OP_అడుగు:
                స్థితి GGML_UNARY_OP_GELU_ERF:
                స్థితి GGML_UNARY_OP_SOFTPLUS:
                    తిరిగి నిజం;
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(op)) {
                స్థితి GGML_GLU_OP_REGLU:
                స్థితి GGML_GLU_OP_GEGLU:
                స్థితి GGML_GLU_OP_SWIGLU:
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                    తిరిగి నిజం;
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_MUL_MAT:
            {
                ఎంపిక (op->src[0]->type) {
#నిర్వచితం_కాకపోతే ASCEND_310P
                    స్థితి GGML_TYPE_BF16:
#అయితే_ముగింపు
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_F32:
                        తిరిగి నిజం;
                    స్థితి GGML_TYPE_Q8_0:
                    స్థితి GGML_TYPE_Q4_0:
#నిర్వచితమైతే ASCEND_310P
                        // Q4 && Q8 per group is not support on 310p device
                        తిరిగి అబద్ధం;
#లేకపోతే
                        // only support contiguous for quantized types.
                        తిరిగి ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#అయితే_ముగింపు
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
            }
        స్థితి GGML_OP_MUL_MAT_ID:
            ఎంపిక (op->src[0]->type) {
                స్థితి GGML_TYPE_F16:
                స్థితి GGML_TYPE_F32:
                    తిరిగి నిజం;
                స్థితి GGML_TYPE_Q8_0:
                స్థితి GGML_TYPE_Q4_0:
#నిర్వచితమైతే ASCEND_310P
                    // Q4 && Q8 per group is not support on 310p device
                    తిరిగి అబద్ధం;
#లేకపోతే
                    // only support contiguous for quantized types.
                    తిరిగి ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#అయితే_ముగింపు
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
        // embedding
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            {
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
#నిర్వచితం_కాకపోతే ASCEND_310P
                    స్థితి GGML_TYPE_BF16:
#అయితే_ముగింపు
                    స్థితి GGML_TYPE_Q8_0:
                        తిరిగి నిజం;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
            }
            ఆపు;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            {
                ఎంపిక (op->type) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
#నిర్వచితం_కాకపోతే ASCEND_310P
                    స్థితి GGML_TYPE_BF16:
#అయితే_ముగింపు
                        తిరిగి నిజం;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
            }
            ఆపు;
        స్థితి GGML_OP_CPY:
            {
                ggml_tensor * src = op->src[0];
#నిర్వచితమైతే ASCEND_310P
                ఒకవేళ ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                    // only support F32 and F16 on 310P.
                    తిరిగి అబద్ధం;
                }
#లేకపోతే
                ఒకవేళ ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
                    // only support F32, F16 and BF16.
                    తిరిగి అబద్ధం;
                }
#అయితే_ముగింపు
                తిరిగి నిజం;
            }
            ఆపు;
        స్థితి GGML_OP_CONT:
            {
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_F16:
#నిర్వచితం_కాకపోతే ASCEND_310P
                    స్థితి GGML_TYPE_BF16:
#అయితే_ముగింపు
                        తిరిగి నిజం;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
            }
        స్థితి GGML_OP_ROPE:
            {
                ఒకవేళ (((స్థిరరాశి int32_t *) op->op_params)[15] != 0) {
                    తిరిగి అబద్ధం; // FIXME: support ggml_rope_set_offset
                }
                ఒకవేళ (op->src[0]->ne[0] > 896) {
                    తిరిగి అబద్ధం;
                }
#నిర్వచితమైతే ASCEND_310P
                // TODO: Support rope_dim < ne00(dim)
                ఒకవేళ (op->src[0]->ne[0] != op->op_params[1]) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (!ggml_is_contiguous(op->src[0])) {
                    తిరిగి అబద్ధం;
                }
#అయితే_ముగింపు
                తిరిగి నిజం;
            }
        స్థితి GGML_OP_UPSCALE:
            {
                // aclnnUpsampleNearest2dGetWorkspaceSize not support
                // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
                ఒకవేళ (op->src[0]->ne[2] * op->ne[3] != op->src[0]->ne[3] * op->ne[2]) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->op_params[0] != GGML_SCALE_బహుళకం_సమీప) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                    తిరిగి అబద్ధం;
                }
                తిరిగి నిజం;
            }
        స్థితి GGML_OP_POOL_2D:
            {
                స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *) op->op_params;
#నిర్వచితమైతే ASCEND_310P
                గణన ggml_op_pool opt = static_cast<ggml_op_pool>(opts[0]);
                ఒకవేళ (opt == GGML_OP_POOL_MAX) {
                    తిరిగి అబద్ధం;
                }
#అయితే_ముగింపు
                స్థిరరాశి పూర్ణాంకం k0 = opts[1];
                స్థిరరాశి పూర్ణాంకం k1 = opts[2];
                స్థిరరాశి పూర్ణాంకం p0 = opts[5];
                స్థిరరాశి పూర్ణాంకం p1 = opts[6];
                // value of paddingH should be at most half of kernelH
                // value of paddingW should be at most half of kernelW
                తిరిగి (p0 <= (k0 / 2)) && (p1 <= (k1 / 2));
            }
        స్థితి GGML_OP_కూడిక:
            తిరిగి ggml_is_contiguous_అడ్డువరుసలు(op->src[0]);
        స్థితి GGML_OP_L2_NORM:
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
        స్థితి GGML_OP_DUP:
        స్థితి GGML_OP_IM2_నిలువు:
        స్థితి GGML_OP_CONCAT:
        స్థితి GGML_OP_REPEAT:
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_NORM:
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_ADD1:
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_DIV:
        స్థితి GGML_OP_RMS_NORM:
        స్థితి GGML_OP_SQR:
        స్థితి GGML_OP_SQRT:
        స్థితి GGML_OP_CLAMP:
        స్థితి GGML_OP_DIAG_ముసుగు_INF:
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
        స్థితి GGML_OP_ARGSORT:
        స్థితి GGML_OP_ACC:
        స్థితి GGML_OP_సమితి:
        స్థితి GGML_OP_సమూహం_NORM:
            తిరిగి నిజం;
        స్థితి GGML_OP_PAD:
            // TODO: add circular padding support for cann, see https://github.com/ggml-org/llama.cpp/pull/16985
            తిరిగి ggml_get_op_params_i32(op, 8) == 0;
        స్థితి GGML_OP_ARANGE:
        స్థితి GGML_OP_TIMESTEP_EMBEDDING:
        స్థితి GGML_OP_LEAKY_RELU:
        స్థితి GGML_OP_ARGMAX:
        స్థితి GGML_OP_COS:
        స్థితి GGML_OP_SIN:
        స్థితి GGML_OP_LOG:
        స్థితి GGML_OP_సగటు:
        స్థితి GGML_OP_PAD_REFLECT_1D:
        స్థితి GGML_OP_లెక్క_EQUAL:
        స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
            తిరిగి నిజం;
        స్థితి GGML_OP_బహిర్గతం_PROD:
            {
#నిర్వచితమైతే ASCEND_310P
                // Ger is not supported on 310p device
                తిరిగి అబద్ధం;
#లేకపోతే
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_F32:
                        తిరిగి నిజం;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
#అయితే_ముగింపు
            }
        స్థితి GGML_OP_CONV_TRANSPOSE_1D:
            తిరిగి నిజం;
        స్థితి GGML_OP_SCALE:
            దశాంశం bias;
            జ్ఞాపకం_నకలు(&bias, (స్థిరరాశి దశాంశం *) (op->op_params) + 1, పరిమాణం(దశాంశం));
            తిరిగి bias == 0.0f;  // TODO: support bias != 0.0f
        స్థితి GGML_OP_SOFT_MAX:
            // TODO: support attention sinks [TAG_ATTN_SINKS]
            ఒకవేళ (op->src[2]) {
                తిరిగి అబద్ధం;
            }
            తిరిగి నిజం;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            {
#నిర్వచితమైతే ASCEND_310P
                // FA not support on 310p device
                తిరిగి అబద్ధం;
#లేకపోతే
                // derived from [ggml-cuda.cu]
                ఒకవేళ (op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32 &&
                    op->src[1]->type != GGML_TYPE_BF16) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16) {
                    తిరిగి అబద్ధం;
                }
                // TODO: support attention sinks [TAG_ATTN_SINKS]
                ఒకవేళ (op->src[4]) {
                    తిరిగి అబద్ధం;
                }
                ఒకవేళ (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                    // different head sizes of K and V are not supported yet
                    తిరిగి అబద్ధం;
                }
                దశాంశం logitSoftcap = 0.0f;
                జ్ఞాపకం_నకలు(&logitSoftcap, (స్థిరరాశి దశాంశం *) (op->op_params) + 2, పరిమాణం(దశాంశం));
                ఒకవేళ (logitSoftcap != 0.0f) {
                    తిరిగి అబద్ధం;
                }
                తిరిగి నిజం;
#అయితే_ముగింపు
            }
        స్థితి GGML_OP_SSM_CONV:
            తిరిగి నిజం;
        స్థితి GGML_OP_CUMSUM:
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_TRI:
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_FILL:
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_DIAG:
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_SOLVE_TRI:
            తిరిగి op->src[0]->type == GGML_TYPE_F32;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    GGML_వాడని(dev);
}

/**
 * @brief Records an event on the CANN backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
స్థిర శూన్య ggml_backend_cann_event_రికార్డు(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrt_రికార్డు_Event((aclrtEvent) event->context, cann_ctx->stream()));
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
స్థిర శూన్య ggml_backend_cann_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ఒకవేళ (ggml_backend_is_cann(backend)) {
        ACL_CHECK(aclrt_ప్రవాహం_Wait_Event(cann_ctx->stream(), (aclrtEvent) event->context));
    } లేకపోతే {
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
స్థిర స్థిరరాశి ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_సమితి_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_cann_లేఖాచిత్రం_compute,
    /* .event_record            = */ ggml_backend_cann_event_రికార్డు,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .graph_optimize          = */ ఖాళీ,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
స్థిర ggml_guid_t ggml_backend_cann_guid() {
    స్థిర ggml_guid guid = { 0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                              0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64 };
    తిరిగి &guid;
}

// backend device
నిర్మాణం ggml_backend_cann_device_context {
    పూర్ణాంకం         device;
    std::string name;
    std::string description;
    పూర్ణాంకం op_offload_min_batch_కొలత;
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    తిరిగి ctx->name.c_str();
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cann_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    తిరిగి ctx->description.c_str();
}

స్థిర శూన్య ggml_backend_cann_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ggml_backend_cann_get_device_జ్ఞాపకం(ctx->device, విడుదల, మొత్తం);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_cann_device_get_type(ggml_backend_dev_t dev) {
    GGML_వాడని(dev);
    తిరిగి GGML_BACKEND_DEVICE_TYPE_GPU;
}

స్థిర శూన్య ggml_backend_cann_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type        = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);

    తార్కిక host_బఫరు = getenv("GGML_CANN_NO_PINNED") == శూన్యసూచిక;

    props->caps = {
        /* .async                 = */ అబద్ధం,
        /* .host_buffer           = */ host_బఫరు,
        /* .buffer_from_host_ptr  = */ అబద్ధం,
        /* .events                = */ నిజం,
        /* .mmap_support          = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    GGML_వాడని(params);
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    తిరిగి ggml_backend_cann_init(ctx->device);
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
స్థిర తార్కిక ggml_backend_cann_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    ఒకవేళ (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context *      dev_ctx  = (ggml_backend_cann_device_context *) dev->context;
        ggml_backend_cann_బఫరు_type_context * buft_ctx = (ggml_backend_cann_బఫరు_type_context *) buft->context;
        తిరిగి buft_ctx->device == dev_ctx->device;
    }
    తిరిగి అబద్ధం;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_cann_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    తిరిగి ggml_backend_cann_బఫరు_type(ctx->device);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_cann_device_get_host_బఫరు_type(ggml_backend_dev_t dev) {
    GGML_వాడని(dev);
    తిరిగి ggml_backend_cann_host_బఫరు_type();
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
స్థిర తార్కిక ggml_backend_cann_offload_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;

    తిరిగి op->ne[1] >= dev_ctx->op_offload_min_batch_కొలత && op->op != GGML_OP_GET_అడ్డువరుసలు;
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
స్థిర ggml_backend_event_t ggml_backend_cann_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *) dev->context;

    ggml_cann_సమితి_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    తిరిగి కొత్త ggml_backend_event{
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
స్థిర శూన్య ggml_backend_cann_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent) event->context));

    తొలగించు event;
    GGML_వాడని(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
స్థిర శూన్య ggml_backend_cann_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent) event->context));

    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_జ్ఞాపకం,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,  // called for every card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_బఫరు_type,
    /* .get_host_buffer_type    = */ ggml_backend_cann_device_get_host_బఫరు_type,
    /* .buffer_from_host_ptr    = */ ఖాళీ,  // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};

// backend reg
నిర్మాణం ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_వాడని(reg);
    తిరిగి GGML_CANN_NAME;
}

స్థిర size_t ggml_backend_cann_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    తిరిగి ctx->devices.size();
}

స్థిర ggml_backend_dev_t ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, size_t సూచీ) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    GGML_ASSERT(సూచీ < ctx->devices.size());
    తిరిగి ctx->devices[సూచీ];
}

స్థిర శూన్య * ggml_backend_cann_reg_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    GGML_వాడని(reg);
    GGML_వాడని(name);
    // reserved for future use
    తిరిగి శూన్యసూచిక;
}

స్థిర స్థిరరాశి ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_లెక్క,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_చిరునామా,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    స్థిర ggml_backend_reg reg;
    స్థిర తార్కిక             initialized = అబద్ధం;

    {
        స్థిర std::mutex           mutex;
        std::lock_guard<std::mutex> తాళం(mutex);
        ఒకవేళ (!initialized) {
            aclInit(శూన్యసూచిక);
            ggml_backend_cann_reg_context * ctx = కొత్త ggml_backend_cann_reg_context;
            స్థిరరాశి పూర్ణాంకం min_batch_కొలత = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            కొరకు (పూర్ణాంకం i = 0; i < ggml_cann_info().device_లెక్క; i++) {
                ggml_backend_cann_device_context * dev_ctx = కొత్త ggml_backend_cann_device_context();
                dev_ctx->description                       = aclrtGetSocName();
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_CANN_NAME + std::to_string(i);
                dev_ctx->op_offload_min_batch_కొలత         = min_batch_కొలత;
                ggml_cann_సమితి_device(i);
                ggml_backend_dev_t dev = కొత్త ggml_backend_device{ /* .iface   = */ ggml_backend_cann_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
                                    /* .iface       = */ ggml_backend_cann_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = నిజం;
    }

    తిరిగి &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(శూన్యసూచిక);
    ఒకవేళ (device < 0 || device >= ggml_backend_cann_get_device_లెక్క()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        తిరిగి శూన్యసూచిక;
    }

    ggml_backend_cann_context * ctx = కొత్త ggml_backend_cann_context(device);
    ఒకవేళ (ctx == శూన్యసూచిక) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        తిరిగి శూన్యసూచిక;
    }
    ggml_cann_సమితి_device(ctx->device);
    ggml_backend_t cann_backend =
        కొత్త ggml_backend{ /* .guid      = */ ggml_backend_cann_guid(),
                          /* .interface = */ ggml_backend_cann_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                          /* .context   = */ ctx };

    తిరిగి cann_backend;
}

తార్కిక ggml_backend_is_cann(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_లెక్క() {
    తిరిగి ggml_cann_info().device_లెక్క;
}

శూన్య ggml_backend_cann_get_device_description(int32_t device, అక్షరం * description, size_t description_కొలత) {
    ggml_cann_సమితి_device(device);
    స్థిరరాశి అక్షరం * soc_name = aclrtGetSocName();
    snprintf(description, description_కొలత, "%s", soc_name);
}

శూన్య ggml_backend_cann_get_device_జ్ఞాపకం(int32_t device, size_t * విడుదల, size_t * మొత్తం) {
    ggml_cann_సమితి_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, విడుదల, మొత్తం));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)

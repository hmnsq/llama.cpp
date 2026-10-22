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

#ಅನ್ತರ್ಭಾವ "ggml-cann.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cann/aclnn_ops.h"
#ಅನ್ತರ್ಭಾವ "ggml-cann/common.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <acl/acl.h>
#ಅನ್ತರ್ಭಾವ <aclnnop/aclnn_trans_matmul_weight.h>
#ಅನ್ತರ್ಭಾವ <stdarg.h>

#ಅನ್ತರ್ಭಾವ <chrono>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <queue>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C

#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] ಶೂನ್ಯ ggml_cann_error(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * stmt, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * func, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * file, ಪೂರ್ಣಾಂಕ ಸಾಲು, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸಂದೇಶ_2) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", ಸಂದೇಶ_2);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, ಸಾಲು);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

// Thread-local variable to record the current device of this thread.
thread_local ಪೂರ್ಣಾಂಕ g_current_cann_device = -1;

/**
 * @brief Set the CANN device to be used.
 *
 * @param device The target device ID to set.
 */
ಶೂನ್ಯ ggml_cann_ಗಣ_device(ಸ್ಥಿರಾಂಕ int32_t device) {
    // int current_device = -1;
    // Note: In some CANN versions, if no device has been set yet,
    //       aclrtGetDevice(&current_device) may return 0 by default.
    // aclrtGetDevice(&current_device);

    // If the current device is already the target one, no need to switch.
    ಆದರೆ (device == g_current_cann_device) {
        ಹಿಂತಿರುಗಿ;
    }

    // Switch to the new device.
    ACL_CHECK(aclrt_ಗಣ_Device(device));

    // Update the global device record.
    g_current_cann_device = device;
}

/**
 * @brief Get the value of the specified environment variable (name) as lowercase.
 *        if not empty, return a std::string object
 */
std::optional<std::string> get_env_as_lowercase(ಸ್ಥಿರಾಂಕ std::string & name) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * val = std::getenv(name.c_str());
    ಆದರೆ (!val) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    std::string res = std::string(val);
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    ಹಿಂತಿರುಗಿ res;
}

/**
 * @brief Verify whether the environment variable is a valid value.
 */
ತಾರ್ಕಿಕ parse_bool(ಸ್ಥಿರಾಂಕ std::string & value) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::unordered_set<std::string> valid_ಮೌಲ್ಯಗಳು = { "on", "1", "yes", "y", "enable", "true" };
    ಹಿಂತಿರುಗಿ valid_ಮೌಲ್ಯಗಳು.find(value) != valid_ಮೌಲ್ಯಗಳು.end();
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
ಪೂರ್ಣಾಂಕ parse_integer(ಸ್ಥಿರಾಂಕ std::string & value) {
    ಪ್ರಯತ್ನಿಸು {
        ಹಿಂತಿರುಗಿ std::stoi(value);
    } ಹಿಡಿ (...) {
        ಹಿಂತಿರುಗಿ 0;
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
ಸ್ಥಿರ ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrt_Get_Device_ಎಣಿಕೆ((uint32_t *) &info.device_ಎಣಿಕೆ);

    ಆದರೆ (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n", __func__, acl_Get_Recent_Err_ಸಂದೇಶ());
        ಹಿಂತಿರುಗಿ info;
    }

    GGML_ASSERT(info.device_ಎಣಿಕೆ <= GGML_CANN_MAX_DEVICES);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id = 0; id < info.device_ಎಣಿಕೆ; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
        prop.ಹಂಚಿಕೆ_Type       = ACL_MEM_ಹಂಚಿಕೆ_TYPE_PINNED;
        prop.memAttr              = ACL_HBM_MEM_HUGE;
        prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = id;
        prop.reserve              = 0;
        err                       = aclrt_Mem_Get_ಹಂಚಿಕೆ_Granularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                                                                     &info.devices[id].vmm_granularity);
        info.devices[id].vmm      = err == ACL_SUCCESS;

        size_t ಬಿಡುಗಡೆ, ಒಟ್ಟು;
        ggml_backend_cann_get_device_ಸ್ಮರಣೆ(id, &ಬಿಡುಗಡೆ, &ಒಟ್ಟು);
        info.devices[id].ಒಟ್ಟು_vram = ಬಿಡುಗಡೆ;
    }

    // TODO: add more device info later.
    ಹಿಂತಿರುಗಿ info;
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
ಸ್ಥಿರಾಂಕ ggml_cann_device_info & ggml_cann_info() {
    ಸ್ಥಿರ ggml_cann_device_info info = ggml_cann_init();
    ಹಿಂತಿರುಗಿ info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(priority segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
ರಚನೆ ggml_cann_pool_buf_prio : ಸಾರ್ವಜನಿಕ ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t alignment = 128;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    ಪೂರ್ಣಾಂಕ device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    ತಾರ್ಕಿಕ disable_clean = ಸುಳ್ಳು;

    /**
     * @brief Structure representing a CANN buffer.
     */
    ರಚನೆ ggml_cann_ಬಫರ್ {
        ಶೂನ್ಯ *                                ptr  = ಶೂನ್ಯಸೂಚಕ;  ///< Pointer to the buffer.
        size_t                                size = 0;        ///< Size of the buffer.
        std::chrono::steady_clock::time_point ಕೊನೆಯ_ಬಳಸಿದ;       ///< Last used time.

        ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ>(ಸ್ಥಿರಾಂಕ ggml_cann_ಬಫರ್ & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ size > other.size; }
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    std::unordered_map<ಶೂನ್ಯ *, size_t>                                                   ಬಫರ್_pool;
    std::priority_queue<ggml_cann_ಬಫರ್, std::vector<ggml_cann_ಬಫರ್>, std::greater<>> free_buffers;

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_ಗಾತ್ರ = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    ಸ್ಪಷ್ಟ ggml_cann_pool_buf_prio(ಪೂರ್ಣಾಂಕ device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf_prio() {
        ggml_cann_ಗಣ_device(device);
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & [b_ptr, b_ಗಾತ್ರ] : ಬಫರ್_pool) {
            aclrtFree(b_ptr);
            pool_ಗಾತ್ರ -= b_ಗಾತ್ರ;
        }
        ಬಫರ್_pool.clear();
        GGML_ASSERT(pool_ಗಾತ್ರ == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    ಶೂನ್ಯ * alloc(size_t size, size_t * actual_ಗಾತ್ರ) override {
        size = GGML_PAD(size, alignment);
        ಆದರೆ (size == 0) {
            size = alignment;
        }

        ಶೂನ್ಯ * ptr = ಶೂನ್ಯಸೂಚಕ;
        ಸ್ವಯಂಚಾಲಿತ   now = std::chrono::steady_clock::now();

        std::vector<ggml_cann_ಬಫರ್> free_buffers_rest;
        free_buffers_rest.reserve(free_buffers.size());
        ಆಗ (!free_buffers.ಖಾಲಿಗಳು()) {
            ಸ್ವಯಂಚಾಲಿತ b = free_buffers.top();
            free_buffers.ತೆಗೆ();

            ಆದರೆ (b.size >= size) {
                // reuse the buffer if the size is enough
                ಸ್ಥಿರಾಂಕ size_t margin = b.size - size;
                ಆದರೆ (margin <= max_reuse_margin) {
                    *actual_ಗಾತ್ರ = b.size;
                    ptr          = b.ptr;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
                    ನಿಲ್ಲಿಸು;
                }
            }

            ತಾರ್ಕಿಕ should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.ಕೊನೆಯ_ಬಳಸಿದ).count() > 100;
            ಆದರೆ (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_ಗಾತ್ರ -= b.size;
                ಬಫರ್_pool.erase(b.ptr);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
                ಮುಂದುವರಿ;
            }
            free_buffers_rest.push_back(b);
        }
        ಗಾಗಿ (ggml_cann_ಬಫರ್ & b : free_buffers_rest) {
            free_buffers.ತಳ್ಳು(std::move(b));
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d] free pool_size = %5u MB\n\n", device,
                      (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (ptr != ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ptr;
        }

        // allocate a new buffer if no buffer can be reused
        ggml_cann_ಗಣ_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_ಮೊದಲ));
        *actual_ಗಾತ್ರ = size;
        pool_ಗಾತ್ರ += size;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: allocate %p, "
            "pool_size = %5u MB, "
            "size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576),
            (uint32_t) (GGML_PAD(size, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
        ಬಫರ್_pool.emplace(ptr, size);
        ಹಿಂತಿರುಗಿ ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size) override {
        GGML_ಬಳಸದ(size);
        ಸ್ವಯಂಚಾಲಿತ it = ಬಫರ್_pool.find(ptr);
        ಆದರೆ (it == ಬಫರ್_pool.end()) {
            GGML_ABORT("cann pool[%d]: buffer %p not found in pool\n", device, ptr);
        }

        ಸ್ವಯಂಚಾಲಿತ now = std::chrono::steady_clock::now();
        free_buffers.emplace(ggml_cann_ಬಫರ್{ ptr, it->second, now });
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: return   %p, "
            "pool_size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
    }
};

/**
 * @brief A pool of CANN buffers(segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
ರಚನೆ ggml_cann_pool_buf : ಸಾರ್ವಜನಿಕ ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    ಪೂರ್ಣಾಂಕ device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    ತಾರ್ಕಿಕ disable_clean = ಸುಳ್ಳು;

    /**
     * @brief Structure representing a CANN buffer.
     */
    ರಚನೆ ggml_cann_ಬಫರ್ {
        ಶೂನ್ಯ *                                ptr  = ಶೂನ್ಯಸೂಚಕ;  ///< Pointer to the buffer memory.
        size_t                                size = 0;        ///< Size of the buffer.
        ತಾರ್ಕಿಕ                                  ಬಳಸಿದ = ಸುಳ್ಳು;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point ಕೊನೆಯ_ಬಳಸಿದ;       ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_ಬಫರ್ ಬಫರ್_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_ಗಾತ್ರ = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    ಸ್ಪಷ್ಟ ggml_cann_pool_buf(ಪೂರ್ಣಾಂಕ device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf() {
        ggml_cann_ಗಣ_device(device);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_ಬಫರ್ & b = ಬಫರ್_pool[i];
            ಆದರೆ (b.ptr != ಶೂನ್ಯಸೂಚಕ) {
                aclrtFree(b.ptr);
                pool_ಗಾತ್ರ -= b.size;
            }
        }
        GGML_ASSERT(pool_ಗಾತ್ರ == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    ಶೂನ್ಯ * alloc(size_t size, size_t * actual_ಗಾತ್ರ) override {
        size = GGML_PAD(size, alignment);
        ಆದರೆ (size == 0) {
            size = alignment;
        }

        ಶೂನ್ಯ * ptr = ಶೂನ್ಯಸೂಚಕ;
        ಸ್ವಯಂಚಾಲಿತ   now = std::chrono::steady_clock::now();

        ಪೂರ್ಣಾಂಕ i = 0;
        ಗಾಗಿ (; i < MAX_BUFFERS; ++i) {
            ggml_cann_ಬಫರ್ & b = ಬಫರ್_pool[i];
            ಆದರೆ (b.ptr == ಶೂನ್ಯಸೂಚಕ) {
                ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (b.ಬಳಸಿದ) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (b.size >= size) {
                // reuse the buffer if the size is enough
                ಸ್ಥಿರಾಂಕ size_t margin = b.size - size;
                ಆದರೆ (margin <= max_reuse_margin) {
                    *actual_ಗಾತ್ರ = b.size;
                    b.ಬಳಸಿದ       = ನಿಜ;
                    ptr          = b.ptr;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
                    ನಿಲ್ಲಿಸು;
                }
            }

            ತಾರ್ಕಿಕ should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.ಕೊನೆಯ_ಬಳಸಿದ).count() > 100;
            ಆದರೆ (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_ಗಾತ್ರ -= b.size;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
                b.ptr = ಶೂನ್ಯಸೂಚಕ;
            }
        }
        ಆದರೆ (ptr != ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ptr;
        }

        ಆದರೆ (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_cann_ಬಫರ್ & b = ಬಫರ್_pool[i];
            ggml_cann_ಗಣ_device(device);
            ACL_CHECK(aclrtMalloc(&b.ptr, size, ACL_MEM_MALLOC_HUGE_ಮೊದಲ));
            pool_ಗಾತ್ರ += size;
            *actual_ಗಾತ್ರ = size;
            b.size       = size;
            b.ಬಳಸಿದ       = ನಿಜ;
            ಆದರೆ (i >= MAX_BUFFERS - 8) {
                GGML_LOG_WARN("cann pool[%d]: slots almost full\n", device);
            }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576),
                (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
            ಹಿಂತಿರುಗಿ b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size) override {
        GGML_ಬಳಸದ(size);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_ಬಫರ್ & b = ಬಫರ್_pool[i];
            ಆದರೆ (b.ptr != ptr) {
                ಮುಂದುವರಿ;
            }
            b.ಬಳಸಿದ      = ಸುಳ್ಳು;
            b.ಕೊನೆಯ_ಬಳಸಿದ = std::chrono::steady_clock::now();
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_ಗಾತ್ರ, 1048576) / 1048576));
#ಆದರೆ_ಅಂತ್ಯ
            ಹಿಂತಿರುಗಿ;
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
ರಚನೆ ggml_cann_pool_vmm : ಸಾರ್ವಜನಿಕ ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    size_t max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    ಪೂರ್ಣಾಂಕ device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    ಶೂನ್ಯ * pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_ಬಳಸಿದ = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_ಗಾತ್ರ = 0;

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
    std::vector<ಶೂನ್ಯ *> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    ಸ್ಪಷ್ಟ ggml_cann_pool_vmm(ಪೂರ್ಣಾಂಕ device) : device(device) {
        ಸ್ವಯಂಚಾಲಿತ dev    = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size    = dev.ಒಟ್ಟು_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        ಆದರೆ (pool_addr != 0) {
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & ಸ್ಥಾನಾಂತರ : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(ಸ್ಥಾನಾಂತರ));
            }
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrt_Release_Mem_ವಿಳಾಸ(pool_addr));
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
    ಶೂನ್ಯ * alloc(size_t size, size_t * actual_ಗಾತ್ರ) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        ಸ್ಥಿರಾಂಕ size_t alignment = 128;
        size                   = GGML_PAD(size, alignment);
        ಆದರೆ (size == 0) {
            size = alignment;
        }

        size_t avail = pool_ಗಾತ್ರ - pool_ಬಳಸಿದ;

        ಆದರೆ (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_ಗಾತ್ರ = size - avail;
            reserve_ಗಾತ್ರ        = GGML_PAD(reserve_ಗಾತ್ರ, granularity);

            GGML_ASSERT(pool_ಗಾತ್ರ + reserve_ಗಾತ್ರ <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
            prop.ಹಂಚಿಕೆ_Type       = ACL_MEM_ಹಂಚಿಕೆ_TYPE_PINNED;
            prop.memAttr              = ACL_HBM_MEM_HUGE;
            prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id          = device;
            prop.reserve              = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_ಗಾತ್ರ, &prop, 0));

            // reserve virtual address space (if not already reserved)
            ಆದರೆ (pool_addr == 0) {
                ACL_CHECK(aclrt_Reserve_Mem_ವಿಳಾಸ(&pool_addr, max_size, 0, ಖಾಲಿ, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((ಅಕ್ಷರ *) pool_addr + pool_ಗಾತ್ರ, reserve_ಗಾತ್ರ, 0, handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((ಅಕ್ಷರ *) pool_addr + pool_ಗಾತ್ರ);

            // add to the pool
            pool_ಗಾತ್ರ += reserve_ಗಾತ್ರ;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
            GGML_LOG_INFO("cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n", device,
                          (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) (pool_ಗಾತ್ರ / 1024 / 1024),
                          (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) (reserve_ಗಾತ್ರ / 1024 / 1024));
#ಆದರೆ_ಅಂತ್ಯ
        }

        GGML_ASSERT(pool_addr != 0);

        ಶೂನ್ಯ * ptr   = (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) pool_addr + pool_ಬಳಸಿದ);
        *actual_ಗಾತ್ರ = size;
        pool_ಬಳಸಿದ += size;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) size,
                      (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ptr);
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size) override {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) size,
                      (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) ptr);
#ಆದರೆ_ಅಂತ್ಯ

        pool_ಬಳಸಿದ -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) pool_addr + pool_ಬಳಸಿದ));
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
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(ಪೂರ್ಣಾಂಕ device) {
    std::string mem_pool_type = get_env_as_lowercase("GGML_CANN_MEM_POOL").value_or("");

    ಆದರೆ (mem_pool_type == "prio") {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        ಹಿಂತಿರುಗಿ std::unique_ptr<ggml_cann_pool>(ಹೊಸ ggml_cann_pool_buf_prio(device));
    }

    ಆದರೆ (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        ಹಿಂತಿರುಗಿ std::unique_ptr<ggml_cann_pool>(ಹೊಸ ggml_cann_pool_vmm(device));
    }

    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    ಹಿಂತಿರುಗಿ std::unique_ptr<ggml_cann_pool>(ಹೊಸ ggml_cann_pool_buf(device));
}

// cann buffer

/**
 * @brief Tracks multi-threaded write progress for a single tensor.
 *
 * When multiple threads call set_tensor on different chunks of the same tensor,
 * this tracker accumulates progress and defers post-processing (quantized format
 * transform or ND-to-NZ conversion) until all data has been written.
 */
ರಚನೆ Tensor_ಗಣ_Tracker {
    std::mutex mtx;                   ///< Protects concurrent access to this tracker
    size_t ಬೈಟುಗಳು_written = 0;         ///< Accumulated bytes written so far
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು = 0;           ///< Target size (full tensor)
    std::vector<uint8_t> host_ಬಫರ್; ///< Host staging buffer for quantized tensors
};

/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
ರಚನೆ ggml_backend_cann_ಬಫರ್_context {
    int32_t device;             ///< The device ID associated with this buffer context.
    ಶೂನ್ಯ *  dev_ptr = ಶೂನ್ಯಸೂಚಕ;  ///< Pointer to the device memory allocated for the buffer.

    std::mutex tracker_mutex;   ///< Protects the trackers map
    std::unordered_map<ಶೂನ್ಯ *, std::unique_ptr<Tensor_ಗಣ_Tracker>> trackers;

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_ಬಫರ್_context(int32_t device, ಶೂನ್ಯ * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_ಬಫರ್_context() { ACL_CHECK(aclrtFree(dev_ptr)); }

    /**
     * @brief Get or create a tracker for the given tensor.
     */
    Tensor_ಗಣ_Tracker * get_or_create_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> ಬೀಗ(tracker_mutex);
        ಸ್ವಯಂಚಾಲಿತ ಕೀಲಿ = tensor->ದತ್ತಾಂಶ;
        ಸ್ವಯಂಚಾಲಿತ it = trackers.find(ಕೀಲಿ);
        ಆದರೆ (it == trackers.end()) {
            ಸ್ವಯಂಚಾಲಿತ tracker = std::make_unique<Tensor_ಗಣ_Tracker>();
            tracker->ಒಟ್ಟು_ಬೈಟುಗಳು = ggml_nbytes(tensor);
            ಸ್ವಯಂಚಾಲಿತ * ptr = tracker.get();
            trackers[ಕೀಲಿ] = std::move(tracker);
            ಹಿಂತಿರುಗಿ ptr;
        }
        ಹಿಂತಿರುಗಿ it->second.get();
    }

    /**
     * @brief Remove the tracker for the given tensor.
     */
    ಶೂನ್ಯ remove_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> ಬೀಗ(tracker_mutex);
        trackers.erase(tensor->ದತ್ತಾಂಶ);
    }
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
ರಚನೆ ggml_backend_cann_ಬಫರ್_type_context {
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
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cann_ಬಫರ್_type_name(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_cann_ಬಫರ್_type_context * buft_ctx = (ggml_backend_cann_ಬಫರ್_type_context *) buft->context;

    ಹಿಂತಿರುಗಿ buft_ctx->name.c_str();
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_buft_is_cann(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ buft->iface.get_name == ggml_backend_cann_ಬಫರ್_type_name;
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_cann_ಬಫರ್_context * ctx = (ggml_backend_cann_ಬಫರ್_context *) ಬಫರ್->context;
    ಅಳಿಸು ctx;
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
ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_cann_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_cann_ಬಫರ್_context * ctx = (ggml_backend_cann_ಬಫರ್_context *) ಬಫರ್->context;
    ಹಿಂತಿರುಗಿ ctx->dev_ptr;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_transform_q4_0(ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, ಶೂನ್ಯ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_ಬೈಟುಗಳು = n_elems * ಗಾತ್ರ(uint8_t) / 2;

    uint8_t *  quant_ಸ್ಥಾನಾಂತರ = (uint8_t *) dst;
    uint16_t * scale_ಸ್ಥಾನಾಂತರ = (uint16_t *) ((ಅಕ್ಷರ *) dst + quant_ಬೈಟುಗಳು);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < groups; i++) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * ಗುಂಪು = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src + i * ಗಾತ್ರ(ಖಂಡ_q4_0));
        *scale_ಸ್ಥಾನಾಂತರ            = ಗುಂಪು->d;
        scale_ಸ್ಥಾನಾಂತರ++;

        // 0-15
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_ಸ್ಥಾನಾಂತರ) = (ಗುಂಪು->qs[j] & 0x0F);
            (*quant_ಸ್ಥಾನಾಂತರ) |= ((ಗುಂಪು->qs[j + 1] << 4));
            quant_ಸ್ಥಾನಾಂತರ++;
        }

        // 16-31
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_ಸ್ಥಾನಾಂತರ) = (ಗುಂಪು->qs[j] >> 4);
            (*quant_ಸ್ಥಾನಾಂತರ) |= (ಗುಂಪು->qs[j + 1] & 0xF0);
            quant_ಸ್ಥಾನಾಂತರ++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    ಗಾಗಿ (quant_ಸ್ಥಾನಾಂತರ = (uint8_t *) dst; quant_ಸ್ಥಾನಾಂತರ < (uint8_t *) dst + quant_ಬೈಟುಗಳು; quant_ಸ್ಥಾನಾಂತರ++) {
        (*quant_ಸ್ಥಾನಾಂತರ) ^= 0x88;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_transform_ಹಿಂಭಾಗ_q4_0(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * src, ಶೂನ್ಯ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_ಬೈಟುಗಳು = n_elems * ಗಾತ್ರ(uint8_t) / 2;

    uint8_t *  quant_ಸ್ಥಾನಾಂತರ = (uint8_t *) src;
    uint16_t * scale_ಸ್ಥಾನಾಂತರ = (uint16_t *) ((ಅಕ್ಷರ *) src + quant_ಬೈಟುಗಳು);

    ಗಾಗಿ (; quant_ಸ್ಥಾನಾಂತರ < (uint8_t *) src + quant_ಬೈಟುಗಳು; quant_ಸ್ಥಾನಾಂತರ++) {
        (*quant_ಸ್ಥಾನಾಂತರ) ^= 0x88;
    }
    quant_ಸ್ಥಾನಾಂತರ = (uint8_t *) src;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < groups; i++) {
        ಖಂಡ_q4_0 * ಗುಂಪು = (ಖಂಡ_q4_0 *) ((ಅಕ್ಷರ *) dst + i * ಗಾತ್ರ(ಖಂಡ_q4_0));
        ಗುಂಪು->d           = *scale_ಸ್ಥಾನಾಂತರ;
        scale_ಸ್ಥಾನಾಂತರ++;

        // 0-15
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 2; j += 2) {
            ಗುಂಪು->qs[j]     = ((*quant_ಸ್ಥಾನಾಂತರ) & 0x0F);
            ಗುಂಪು->qs[j + 1] = ((*quant_ಸ್ಥಾನಾಂತರ) >> 4);
            quant_ಸ್ಥಾನಾಂತರ++;
        }

        // 16-31
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 2; j += 2) {
            ಗುಂಪು->qs[j] |= ((*quant_ಸ್ಥಾನಾಂತರ) << 4);
            ಗುಂಪು->qs[j + 1] |= ((*quant_ಸ್ಥಾನಾಂತರ) & 0xF0);
            quant_ಸ್ಥಾನಾಂತರ++;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_transform_q8_0(ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, ಶೂನ್ಯ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_ಬೈಟುಗಳು = n_elems * ಗಾತ್ರ(uint8_t);

    uint8_t *  quant_ಸ್ಥಾನಾಂತರ = (uint8_t *) dst;
    uint16_t * scale_ಸ್ಥಾನಾಂತರ = (uint16_t *) ((ಅಕ್ಷರ *) dst + quant_ಬೈಟುಗಳು);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < groups; i++) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * ಗುಂಪು = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src + i * ಗಾತ್ರ(ಖಂಡ_q8_0));
        *scale_ಸ್ಥಾನಾಂತರ            = ಗುಂಪು->d;
        scale_ಸ್ಥಾನಾಂತರ++;
        size_t ಗುಂಪು_quant_ಗಾತ್ರ = QK8_0 * ಗಾತ್ರ(uint8_t);
        ಸ್ಮರಣೆ_ನಕಲು(quant_ಸ್ಥಾನಾಂತರ, ಗುಂಪು->qs, ಗುಂಪು_quant_ಗಾತ್ರ);
        quant_ಸ್ಥಾನಾಂತರ += ಗುಂಪು_quant_ಗಾತ್ರ;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_transform_ಹಿಂಭಾಗ_q8_0(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, ಶೂನ್ಯ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_ಬೈಟುಗಳು = n_elems * ಗಾತ್ರ(uint8_t);

    ಸ್ಥಿರಾಂಕ uint8_t *  quant_ಸ್ಥಾನಾಂತರ = (ಸ್ಥಿರಾಂಕ uint8_t *) src;
    ಸ್ಥಿರಾಂಕ uint16_t * scale_ಸ್ಥಾನಾಂತರ = (ಸ್ಥಿರಾಂಕ uint16_t *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src + quant_ಬೈಟುಗಳು);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < groups; i++) {
        ಖಂಡ_q8_0 * ಗುಂಪು = (ಖಂಡ_q8_0 *) ((ಅಕ್ಷರ *) dst + i * ಗಾತ್ರ(ಖಂಡ_q8_0));
        ಗುಂಪು->d           = *scale_ಸ್ಥಾನಾಂತರ;
        scale_ಸ್ಥಾನಾಂತರ++;
        size_t ಗುಂಪು_quant_ಗಾತ್ರ = QK8_0 * ಗಾತ್ರ(uint8_t);
        ಸ್ಮರಣೆ_ನಕಲು(ಗುಂಪು->qs, quant_ಸ್ಥಾನಾಂತರ, ಗುಂಪು_quant_ಗಾತ್ರ);
        quant_ಸ್ಥಾನಾಂತರ += ಗುಂಪು_quant_ಗಾತ್ರ;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_transform(ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, ಶೂನ್ಯ * dst) {
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_transform_ಹಿಂಭಾಗ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * src, ಶೂನ್ಯ * dst) {
    ಆಯ್ಕೆ (tensor->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_ಹಿಂಭಾಗ_q4_0(tensor, src, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_ಹಿಂಭಾಗ_q8_0(tensor, src, dst);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ need_transform(ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
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
ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_cann_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor) {
    ಆದರೆ (tensor->view_src != ಖಾಲಿ && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->ಬಫರ್->buft == ಬಫರ್->buft);
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
    }

    // TODO: cann backend doesn't support quantized yet. Just leave the code
    // here.
    ಆದರೆ (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_ಗಾತ್ರ = ggml_nbytes(tensor);
        size_t padded_ಗಾತ್ರ   = ggml_backend_buft_get_alloc_ಗಾತ್ರ(ಬಫರ್->buft, tensor);

        ಆದರೆ (padded_ಗಾತ್ರ > original_ಗಾತ್ರ && tensor->view_src == ಶೂನ್ಯಸೂಚಕ) {
            size_t memset_ಗಾತ್ರ = padded_ಗಾತ್ರ - original_ಗಾತ್ರ;
            ACL_CHECK(aclrtMemset((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + original_ಗಾತ್ರ, memset_ಗಾತ್ರ, 0, memset_ಗಾತ್ರ));
        }
    }
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

/**
 * @brief Workspace for caching NZ buffers per device.
 *
 * This struct manages a device buffer used in NZ computations. It supports
 * allocation, reallocation, and clearing of cached memory. The struct is
 * designed to be used with a global array, one per device.
 */
ರಚನೆ ggml_cann_nz_workspace {
    std::mutex mtx;    // Protects ptr/allocated from concurrent access
    ಶೂನ್ಯ * ptr;        // Pointer to allocated device buffer
    size_t allocated;  // Size of currently allocated buffer in bytes

    /**
     * @brief Constructor. Initializes the workspace with no allocated memory.
     */
    ggml_cann_nz_workspace() : ptr(ಶೂನ್ಯಸೂಚಕ), allocated(0) {}

    /**
     * @brief Free cached memory and reset the workspace.
     *
     * If a buffer has been allocated, this function releases it using
     * aclrtFree and resets internal state.
     */
    ಶೂನ್ಯ clear() {
        ಆದರೆ (ptr) {
            ACL_CHECK(aclrtFree(ptr));
            ptr       = ಶೂನ್ಯಸೂಚಕ;
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
    ಶೂನ್ಯ realloc(size_t new_ಗಾತ್ರ) {
        ಆದರೆ (new_ಗಾತ್ರ > allocated) {
            clear();
            ACL_CHECK(aclrtMalloc(&ptr, new_ಗಾತ್ರ, ACL_MEM_MALLOC_HUGE_ಮೊದಲ));
            allocated = new_ಗಾತ್ರ;
        }
    }

    /**
     * @brief Get the device buffer pointer.
     *
     * @return Pointer to the allocated buffer, or nullptr if not allocated.
     */
    ಶೂನ್ಯ * get() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ptr; }
};

/**
 * @brief Global array of NZ workspaces, one per device.
 */
ಸ್ಥಿರ ggml_cann_nz_workspace g_nz_workspaces[GGML_CANN_MAX_DEVICES];

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
ಸ್ಥಿರ ಶೂನ್ಯ ತೂಕ_format_ಗಮ್ಯ_nz(ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ device) {
    acl_tensor_ptr ತೂಕ_Transposed = ggml_cann_create_tensor(tensor, tensor->ne, tensor->nb, 2, ACL_FORMAT_ND, 0);
    uint64_t       workspace_ಗಾತ್ರ    = 0;
    aclOpExecutor * executor;

    // TransMatmulWeight
    ACL_CHECK(aclnn_Trans_Matmul_ತೂಕ_Get_Workspace_ಗಾತ್ರ(ತೂಕ_Transposed.get(), &workspace_ಗಾತ್ರ, &executor));

    std::lock_guard<std::mutex> ಬೀಗ(g_nz_workspaces[device].mtx);
    // Avoid frequent malloc/free of the workspace.
    g_nz_workspaces[device].realloc(workspace_ಗಾತ್ರ);

    ಶೂನ್ಯ * g_nz_workspace = g_nz_workspaces[device].get();

    ACL_CHECK(aclnn_Trans_Matmul_ತೂಕ(g_nz_workspace, workspace_ಗಾತ್ರ, executor, ಶೂನ್ಯಸೂಚಕ));
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                ggml_tensor *         tensor,
                                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                                size_t                ಸ್ಥಾನಾಂತರ,
                                                size_t                size) {
    ggml_backend_cann_ಬಫರ್_context * ctx = (ggml_backend_cann_ಬಫರ್_context *) ಬಫರ್->context;

    ggml_cann_ಗಣ_device(ctx->device);

    // Only check env once.
    ಸ್ಥಿರ ತಾರ್ಕಿಕ ತೂಕ_ಗಮ್ಯ_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    ತಾರ್ಕಿಕ is_quantized = need_transform(tensor->type);
    ತಾರ್ಕಿಕ is_nz        = !is_quantized && tensor->type != GGML_TYPE_BF16 && ತೂಕ_ಗಮ್ಯ_nz &&
                 is_matmul_ತೂಕ((ಸ್ಥಿರಾಂಕ ggml_tensor *) tensor);

    // Plain tensor (not quantized, not NZ): direct copy, no tracking needed
    ಆದರೆ (!is_quantized && !is_nz) {
        ACL_CHECK(aclrtMemcpy((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size, ದತ್ತಾಂಶ, size, ACL_MEMCPY_HOST_ಗಮ್ಯ_DEVICE));
        ಹಿಂತಿರುಗಿ;
    }

    // Single-shot write (full tensor at once): handle directly without tracking overhead
    ಆದರೆ (ಸ್ಥಾನಾಂತರ == 0 && size == ggml_nbytes(tensor)) {
        ಆದರೆ (is_quantized) {
            ಶೂನ್ಯ * transform_ಬಫರ್ = ಸ್ಮರಣೆ_ಕೊಡು(size);
            ggml_backend_cann_transform(tensor, ದತ್ತಾಂಶ, transform_ಬಫರ್);
            ACL_CHECK(aclrtMemcpy(tensor->ದತ್ತಾಂಶ, size, transform_ಬಫರ್, size, ACL_MEMCPY_HOST_ಗಮ್ಯ_DEVICE));
            ಬಿಡುಗಡೆ(transform_ಬಫರ್);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // NZ weight
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ACL_CHECK(aclrtMemcpy(tensor->ದತ್ತಾಂಶ, size, ದತ್ತಾಂಶ, size, ACL_MEMCPY_HOST_ಗಮ್ಯ_DEVICE));
            ತೂಕ_format_ಗಮ್ಯ_nz(tensor, ctx->device);
        }
        ಹಿಂತಿರುಗಿ;
    }

    // Chunked write: use tracker to accumulate progress and defer transform/conversion
    Tensor_ಗಣ_Tracker * tracker = ctx->get_or_create_tracker(tensor);
    std::unique_lock<std::mutex> ಬೀಗ(tracker->mtx);

    ಆದರೆ (is_quantized) {
        // Stage data in host buffer; transform requires full tensor data
        ಆದರೆ (tracker->host_ಬಫರ್.ಖಾಲಿಗಳು()) {
            tracker->host_ಬಫರ್.resize(tracker->ಒಟ್ಟು_ಬೈಟುಗಳು);
        }
        ಸ್ಮರಣೆ_ನಕಲು(tracker->host_ಬಫರ್.ದತ್ತಾಂಶ() + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // NZ weight: upload chunk to device immediately, defer conversion
        ACL_CHECK(aclrtMemcpy((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size, ದತ್ತಾಂಶ, size, ACL_MEMCPY_HOST_ಗಮ್ಯ_DEVICE));
    }

    tracker->ಬೈಟುಗಳು_written += size;

    // All chunks received: perform deferred transform/conversion
    ಆದರೆ (tracker->ಬೈಟುಗಳು_written >= tracker->ಒಟ್ಟು_ಬೈಟುಗಳು) {
        ಆದರೆ (is_quantized) {
            ಶೂನ್ಯ * transform_ಬಫರ್ = ಸ್ಮರಣೆ_ಕೊಡು(tracker->ಒಟ್ಟು_ಬೈಟುಗಳು);
            ggml_backend_cann_transform(tensor, tracker->host_ಬಫರ್.ದತ್ತಾಂಶ(), transform_ಬಫರ್);
            ACL_CHECK(aclrtMemcpy(tensor->ದತ್ತಾಂಶ, tracker->ಒಟ್ಟು_ಬೈಟುಗಳು, transform_ಬಫರ್, tracker->ಒಟ್ಟು_ಬೈಟುಗಳು, ACL_MEMCPY_HOST_ಗಮ್ಯ_DEVICE));
            ಬಿಡುಗಡೆ(transform_ಬಫರ್);
        }

        ಆದರೆ (is_nz) {
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ತೂಕ_format_ಗಮ್ಯ_nz(tensor, ctx->device);
        }

        // Unlock before removing tracker, as remove_tracker destroys the mutex
        ಬೀಗ.unlock();
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                ಸ್ಥಿರಾಂಕ ggml_tensor *   tensor,
                                                ಶೂನ್ಯ *                ದತ್ತಾಂಶ,
                                                size_t                ಸ್ಥಾನಾಂತರ,
                                                size_t                size) {
    ggml_backend_cann_ಬಫರ್_context * ctx = (ggml_backend_cann_ಬಫರ್_context *) ಬಫರ್->context;

    ggml_cann_ಗಣ_device(ctx->device);

    ಆದರೆ (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(ದತ್ತಾಂಶ, size, (ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size, ACL_MEMCPY_DEVICE_ಗಮ್ಯ_HOST));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಶೂನ್ಯ * transform_ಬಫರ್ = ಸ್ಮರಣೆ_ಕೊಡು(size);
        ACL_CHECK(aclrtMemcpy(transform_ಬಫರ್, size, (ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size, ACL_MEMCPY_DEVICE_ಗಮ್ಯ_HOST));
        ggml_backend_cann_transform_ಹಿಂಭಾಗ(tensor, transform_ಬಫರ್, ದತ್ತಾಂಶ);
        ಬಿಡುಗಡೆ(transform_ಬಫರ್);
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cann_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                ಸ್ಥಿರಾಂಕ ggml_tensor *   src,
                                                ggml_tensor *         dst) {
    ಆದರೆ (ggml_backend_buft_is_cann(src->ಬಫರ್->buft)) {
        ggml_backend_cann_ಬಫರ್_context * src_ctx = (ggml_backend_cann_ಬಫರ್_context *) src->ಬಫರ್->context;
        ggml_backend_cann_ಬಫರ್_context * dst_ctx = (ggml_backend_cann_ಬಫರ್_context *) ಬಫರ್->context;

        size_t memcpy_ಗಾತ್ರ = ggml_nbytes(src);
        // Same device.
        ಆದರೆ (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ, memcpy_ಗಾತ್ರ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src->ದತ್ತಾಂಶ, memcpy_ಗಾತ್ರ,
                                  ACL_MEMCPY_DEVICE_ಗಮ್ಯ_DEVICE));
            ಹಿಂತಿರುಗಿ ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
            // TODO: Support 310p P2P copy
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device, dst_ctx->device));
            ಆದರೆ (canAccessPeer) {
                ggml_cann_ಗಣ_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ, memcpy_ಗಾತ್ರ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src->ದತ್ತಾಂಶ, memcpy_ಗಾತ್ರ,
                                      ACL_MEMCPY_DEVICE_ಗಮ್ಯ_DEVICE));
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_backend_cann_ಬಫರ್_context * ctx = (ggml_backend_cann_ಬಫರ್_context *) ಬಫರ್->context;

    ggml_cann_ಗಣ_device(ctx->device);
    ACL_CHECK(aclrtMemset((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size, value, size));
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ggml_backend_cann_ಬಫರ್_context * ctx = (ggml_backend_cann_ಬಫರ್_context *) ಬಫರ್->context;

    ggml_cann_ಗಣ_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, ಬಫರ್->size, value, ಬಫರ್->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_i ggml_backend_cann_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_cann_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_cann_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_cann_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cann_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cann_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_cann_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ggml_backend_cann_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
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
ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_cann_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ggml_backend_cann_ಬಫರ್_type_context * buft_ctx = (ggml_backend_cann_ಬಫರ್_type_context *) buft->context;

    ggml_cann_ಗಣ_device(buft_ctx->device);

    ಸ್ಥಿರಾಂಕ size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    ಆದರೆ (size == 0) {
        size = alignment;
    }
    ಶೂನ್ಯ *   dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_ಮೊದಲ);
    ಆದರೆ (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, buft_ctx->device, acl_Get_Recent_Err_ಸಂದೇಶ());
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ggml_backend_cann_ಬಫರ್_context * ctx = ಹೊಸ ggml_backend_cann_ಬಫರ್_context(buft_ctx->device, dev_ptr);

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_cann_ಬಫರ್_interface, ctx, size);
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
ಸ್ಥಿರ size_t ggml_backend_cann_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ 128;

    GGML_ಬಳಸದ(buft);
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
ಸ್ಥಿರ size_t ggml_backend_cann_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft,
                                                           ಸ್ಥಿರಾಂಕ ggml_tensor *        tensor) {
    size_t  size = ggml_nbytes(tensor);
    int64_t ne0  = tensor->ne[0];

    // Only check env once.
    ಸ್ಥಿರ ತಾರ್ಕಿಕ ತೂಕ_ಗಮ್ಯ_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);
    ಆದರೆ (ggml_is_quantized(tensor->type)) {
        ಆದರೆ (ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING != 0) {
            size += ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING - ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ತೂಕ_ಗಮ್ಯ_nz && tensor->type != GGML_TYPE_BF16
               && is_matmul_ತೂಕ((ಸ್ಥಿರಾಂಕ ggml_tensor *) tensor)) {
        // NZ format weight are not support quantized yet.
        // If ND tensor transform to NZ, size may changed.
        int64_t shape[] = { tensor->ne[1], tensor->ne[0] };
        GGML_ASSERT(tensor->ne[2] == 1);
        GGML_ASSERT(tensor->ne[3] == 1);
        ಸ್ಥಿರಾಂಕ aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
        size_t              new_ಗಾತ್ರ;
        ACL_CHECK(aclnn_Calculate_Matmul_ತೂಕ_ಗಾತ್ರ_V2(acl_shape, ggml_cann_type_mapping(tensor->type), &new_ಗಾತ್ರ));
        ACL_CHECK(aclDestroyIntArray(acl_shape));
        size = std::max(size, new_ಗಾತ್ರ);
    }

    ಹಿಂತಿರುಗಿ size;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cann_ಬಫರ್_type_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    GGML_ಬಳಸದ(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_type_i ggml_backend_cann_ಬಫರ್_type_interface = {
    /* .get_name            = */ ggml_backend_cann_ಬಫರ್_type_name,
    /* .alloc_buffer        = */ ggml_backend_cann_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ಖಾಲಿ,
    /* .get_alignment       = */ ggml_backend_cann_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ಖಾಲಿ,  // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cann_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ggml_backend_cann_ಬಫರ್_type_is_host,
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
ggml_backend_ಬಫರ್_type_t ggml_backend_cann_ಬಫರ್_type(int32_t device) {
    ಸ್ಥಿರ std::mutex           mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);

    ಆದರೆ (device >= ggml_backend_cann_get_device_ಎಣಿಕೆ()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಸ್ಥಿರ ggml_backend_ಬಫರ್_type ggml_backend_cann_ಬಫರ್_types[GGML_CANN_MAX_DEVICES];

    ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cann_ಬಫರ್_type_initialized = ಸುಳ್ಳು;

    ಆದರೆ (!ggml_backend_cann_ಬಫರ್_type_initialized) {
        ಗಾಗಿ (int32_t i = 0; i < ggml_cann_info().device_ಎಣಿಕೆ; i++) {
            ggml_backend_cann_ಬಫರ್_types[i] = {
                /* .iface    = */ ggml_backend_cann_ಬಫರ್_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), i),
                /* .context  = */
                ಹೊಸ ggml_backend_cann_ಬಫರ್_type_context{ i, "CANN" + std::to_string(i) },
            };
        }
        ggml_backend_cann_ಬಫರ್_type_initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ &ggml_backend_cann_ಬಫರ್_types[device];
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
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cann_host_ಬಫರ್_type_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ "CANN_Host";

    GGML_ಬಳಸದ(buft);
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
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cann_host_ಬಫರ್_name(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ "CANN_Host";

    GGML_ಬಳಸದ(ಬಫರ್);
}

/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer, including
 * its context.
 *
 * @param buffer The CANN host buffer to free.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_host_ಬಫರ್_free(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ACL_CHECK(aclrtFreeHost(ಬಫರ್->context));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
ಸ್ಥಿರ ಶೂನ್ಯ * ggml_cann_host_malloc(size_t size) {
    ಆದರೆ (getenv("GGML_CANN_NO_PINNED") != ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಸ್ಥಿರಾಂಕ size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    ಆದರೆ (size == 0) {
        size = alignment;
    }

    ಶೂನ್ಯ *   hostPtr = ಶೂನ್ಯಸೂಚಕ;
    aclError err     = aclrtMallocHost((ಶೂನ್ಯ **) &hostPtr, size);
    ಆದರೆ (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                      acl_Get_Recent_Err_ಸಂದೇಶ());
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_cann_host_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft,
                                                                             size_t                     size) {
    ಶೂನ್ಯ * hostPtr = ggml_cann_host_malloc(size);

    ಆದರೆ (hostPtr == ಶೂನ್ಯಸೂಚಕ) {
        // fallback to cpu buffer
        ಹಿಂತಿರುಗಿ ggml_backend_buft_alloc_ಬಫರ್(ggml_backend_cpu_ಬಫರ್_type(), size);
    }

    ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_cpu_ಬಫರ್_from_ptr(hostPtr, size);
    ಬಫರ್->buft                 = buft;
    ಬಫರ್->iface.free_ಬಫರ್    = ggml_backend_cann_host_ಬಫರ್_free;

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_ಬಫರ್_type_t ggml_backend_cann_host_ಬಫರ್_type() {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_cann_ಬಫರ್_type_host = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cann_host_ಬಫರ್_type_name,
                           /* .alloc_buffer         = */ ggml_backend_cann_host_ಬಫರ್_type_alloc_ಬಫರ್,
                           /* .alloc_buffer_n       = */ ಖಾಲಿ,
                           /* .get_alignment        = */ ggml_backend_cpu_ಬಫರ್_type()->iface.get_alignment,
                           /* .get_max_size         = */ ಖಾಲಿ,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_ಬಫರ್_type()->iface.get_alloc_ಗಾತ್ರ,
                           /* .get_alloc_size_n     = */ ಖಾಲಿ,
                           /* .is_host              = */ ggml_backend_cpu_ಬಫರ್_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
        /* .context  = */ ಶೂನ್ಯಸೂಚಕ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_cann_ಬಫರ್_type_host;
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_cann_compute_forward(ggml_backend_cann_context & ctx, ರಚನೆ ggml_tensor * dst) {
    ಆಯ್ಕೆ (dst->op) {
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ggml_cann_get_ಅಡ್ಡಸಾಲುಗಳು(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            ggml_cann_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_ADD1:
            ggml_cann_ದ್ವಿಮಾನ_op<aclnn_ಕೂಡಿಸು>(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SUB:
            ggml_cann_ದ್ವಿಮಾನ_op<aclnn_sub>(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL:
            ggml_cann_ದ್ವಿಮಾನ_op<aclnn_mul>(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIV:
            ggml_cann_ದ್ವಿಮಾನ_op<aclnn_div>(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(dst)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_OP_UNARY(Abs);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_OP_UNARY(Neg);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                    // aclnnGelu internally uses the erf-based approximation.
                    GGML_CANN_CALL_OP_UNARY(Gelu);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_OP_UNARY(Silu);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                    {
                        ಸ್ವಯಂಚಾಲಿತ lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                        };
                        ggml_cann_op_unary(lambda, ctx, dst);
                    }
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                    GGML_CANN_CALL_OP_UNARY(Tanh);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                    GGML_CANN_CALL_OP_UNARY(Relu);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Sigmoid);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Hardsigmoid);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                    GGML_CANN_CALL_OP_UNARY(Hardswish);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                    GGML_CANN_CALL_OP_UNARY(Exp);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                    ggml_cann_elu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                    GGML_CANN_CALL_OP_UNARY(Sign);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                    ggml_cann_ಹೆಜ್ಜೆ(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                    ggml_cann_softplus(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(dst)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                    GGML_CANN_CALL_OP_UNARY_GATED(Relu);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                    ggml_cann_geglu(ctx, dst, 0);  // approximate=0 → tanh
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                    ggml_cann_geglu(ctx, dst, 1);  // approximate=1 → erf
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                    ggml_cann_swiglu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cann_swiglu_clamp(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                    ggml_cann_geglu_ಶೀಘ್ರ(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
            ggml_cann_ಗುಂಪು_norm(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            ggml_cann_l2_norm(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cann_cross_entropy_loss(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == ಶೂನ್ಯಸೂಚಕ);
            dst->src[1] = dst->src[0];
            ggml_cann_ದ್ವಿಮಾನ_op<aclnn_mul>(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQRT:
            GGML_CANN_CALL_OP_UNARY(Sqrt);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
            ggml_cann_ಗಣ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
            ggml_cann_diag_ಮುಖವಾಡ(ctx, dst, -INFINITY);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            ggml_cann_im2col(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
            ggml_cann_ಮೊತ್ತ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            ggml_cann_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            ggml_cann_argmax(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_COS:
            ggml_cann_op_unary<aclnn_cos>(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SIN:
            ggml_cann_op_unary<aclnn_sin>(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cann_conv_transpose_1d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LOG:
            GGML_CANN_CALL_OP_UNARY(Log);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            ggml_cann_ಸರಾಸರಿ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
            ggml_cann_pad_reflect_1d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
            ggml_cann_ಎಣಿಕೆ_equal(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
            ggml_cann_ಹೊರಹೋಗು_prod(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
            ggml_cann_gated_ರೇಖೀಯ_attn(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            ggml_cann_ssm_conv(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            ggml_cann_cumsum(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TRI:
            ggml_cann_tri(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FILL:
            ggml_cann_fill(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG:
            ggml_cann_diag(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            ggml_cann_solve_tri(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
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
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;

    ಹಿಂತಿರುಗಿ cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    ಅಳಿಸು cann_ctx;
    ಅಳಿಸು backend;
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_ಗಣ_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *  tensor,
                                               ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *   ದತ್ತಾಂಶ,
                                               size_t         ಸ್ಥಾನಾಂತರ,
                                               size_t         size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_ಬಫರ್_t       buf      = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;

    GGML_ASSERT(buf->buft == ggml_backend_cann_ಬಫರ್_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size, ದತ್ತಾಂಶ, size, ACL_MEMCPY_HOST_ಗಮ್ಯ_DEVICE,
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_get_tensor_async(ggml_backend_t      backend,
                                               ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                                               ಶೂನ್ಯ *              ದತ್ತಾಂಶ,
                                               size_t              ಸ್ಥಾನಾಂತರ,
                                               size_t              size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_ಬಫರ್_t       buf      = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;

    GGML_ASSERT(buf->buft == ggml_backend_cann_ಬಫರ್_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync(ದತ್ತಾಂಶ, size, (ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size, ACL_MEMCPY_DEVICE_ಗಮ್ಯ_HOST,
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cann_cpy_tensor_async(ggml_backend_t      backend_src,
                                               ggml_backend_t      backend_dst,
                                               ಸ್ಥಿರಾಂಕ ggml_tensor * src,
                                               ggml_tensor *       dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) || ggml_backend_is_cann(backend_dst));

    GGML_ASSERT(!is_matmul_ತೂಕ((ಸ್ಥಿರಾಂಕ ggml_tensor *) src));

    ಆದರೆ (!ggml_backend_buft_is_cann(src->ಬಫರ್->buft) || !ggml_backend_buft_is_cann(dst->ಬಫರ್->buft)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_backend_ಬಫರ್_t buf_src = src->view_src ? src->view_src->ಬಫರ್ : src->ಬಫರ್;
    ggml_backend_ಬಫರ್_t buf_dst = dst->view_src ? dst->view_src->ಬಫರ್ : dst->ಬಫರ್;

    ggml_backend_cann_context * cann_ctx_src = (ggml_backend_cann_context *) backend_src->context;
    ggml_backend_cann_context * cann_ctx_dst = (ggml_backend_cann_context *) backend_dst->context;

    size_t copy_ಗಾತ್ರ = ggml_nbytes(dst);
    ಆದರೆ (copy_ಗಾತ್ರ == 0) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಆದರೆ (backend_src != backend_dst) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
        // TODO: Support 310p P2P copy
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
        ggml_backend_cann_ಬಫರ್_context * buf_ctx_src = (ggml_backend_cann_ಬಫರ್_context *) buf_src->context;
        ggml_backend_cann_ಬಫರ್_context * buf_ctx_dst = (ggml_backend_cann_ಬಫರ್_context *) buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device, cann_ctx_dst->device));
        ಆದರೆ (!canAccessPeer) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // need open both directions for memcpyasync between devices.
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_ಗಣ_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        // wait for task_queue empty to keep task order.
        ACL_CHECK(aclrtMemcpyAsync(dst->ದತ್ತಾಂಶ, copy_ಗಾತ್ರ, src->ದತ್ತಾಂಶ, copy_ಗಾತ್ರ, ACL_MEMCPY_DEVICE_ಗಮ್ಯ_DEVICE,
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
        ACL_CHECK(aclrt_Synchronize_ಹರಿವು(cann_ctx_src->stream()));
    } ಇಲ್ಲದಿದ್ದರೆ {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->ದತ್ತಾಂಶ, copy_ಗಾತ್ರ, src->ದತ್ತಾಂಶ, copy_ಗಾತ್ರ, ACL_MEMCPY_DEVICE_ಗಮ್ಯ_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_ಗಣ_device(cann_ctx->device);
    ACL_CHECK(aclrt_Synchronize_ಹರಿವು(cann_ctx->stream()));
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_cann_can_fuse(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph *          cgraph,
                               ಪೂರ್ಣಾಂಕ                                 ಗಂಟು_idx,
                               std::initializer_list<ಎಣಿಕೆ ggml_op> ops) {
    ಆದರೆ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, ops)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // CANN backend supports fusing ADD + RMS_NORM operations
    ಆದರೆ ((ops.size() == 2) && ops.begin()[0] == GGML_OP_ಕೂಡಿಸು && ops.begin()[1] == GGML_OP_RMS_NORM) {
        ggml_tensor * ಕೂಡಿಸು_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
        // TODO: support broadcast for ADD + RMS_NORM
        ಆದರೆ (ಕೂಡಿಸು_ಗಂಟು->src[0]->ne[0] != ಕೂಡಿಸು_ಗಂಟು->src[1]->ne[0] || ಕೂಡಿಸು_ಗಂಟು->src[0]->ne[1] != ಕೂಡಿಸು_ಗಂಟು->src[1]->ne[1] ||
            ಕೂಡಿಸು_ಗಂಟು->src[0]->ne[2] != ಕೂಡಿಸು_ಗಂಟು->src[1]->ne[2] || ಕೂಡಿಸು_ಗಂಟು->src[0]->ne[3] != ಕೂಡಿಸು_ಗಂಟು->src[1]->ne[3]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
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
ಸ್ಥಿರ ಶೂನ್ಯ evaluate_and_capture_cann_ನಕ್ಷೆ(ggml_backend_cann_context * cann_ctx,
                                            ggml_cgraph *               cgraph,
                                            ತಾರ್ಕಿಕ                        use_cann_ನಕ್ಷೆ,
                                            ತಾರ್ಕಿಕ                        cann_ನಕ್ಷೆ_capture_required) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ USE_ACL_ನಕ್ಷೆ
    ಆದರೆ (use_cann_ನಕ್ಷೆ && cann_ನಕ್ಷೆ_capture_required) {  // Begin CANN graph capture
        ACL_CHECK(aclmdlRICaptureBegin(cann_ctx->stream(), ACL_MODEL_RI_CAPTURE_ಬಹುಲಕ_GLOBAL));
    }
#ಆದರೆ_ಅಂತ್ಯ  // USE_ACL_GRAPH
    // Only perform the graph execution if CANN graphs are not enabled, or we are capturing the graph.
    // With the use of CANN graphs, the execution will be performed by the graph launch.
    ಸ್ಥಿರ ತಾರ್ಕಿಕ opt_fusion = parse_bool(get_env_as_lowercase("GGML_CANN_OPERATOR_FUSION").value_or(""));

    ಆದರೆ (!use_cann_ನಕ್ಷೆ || cann_ನಕ್ಷೆ_capture_required) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * ಗಂಟು = cgraph->nodes[i];
            ಆದರೆ (opt_fusion) {
                ಆದರೆ (ggml_cann_can_fuse(cgraph, i, { GGML_OP_ಕೂಡಿಸು, GGML_OP_RMS_NORM })) {
                    ggml_cann_op_ಕೂಡಿಸು_rms_norm_fused(*cann_ctx, ಗಂಟು, cgraph->nodes[i + 1]);
                    i++;
                    ಮುಂದುವರಿ;
                }
            }

            ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(ಗಂಟು) || ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_TRANSPOSE ||
                ಗಂಟು->op == GGML_OP_VIEW || ಗಂಟು->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ || ಗಂಟು->op == GGML_OP_NONE) {
                ಮುಂದುವರಿ;
            }

            ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ಮುಂದುವರಿ;
            }

            ತಾರ್ಕಿಕ ok = ggml_cann_compute_forward(*cann_ctx, ಗಂಟು);
            ಆದರೆ (!ok) {
                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, ಗಂಟು->name, ggml_op_name(ಗಂಟು->op));
            }
            GGML_ASSERT(ok);
        }
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ USE_ACL_ನಕ್ಷೆ
    ಆದರೆ (use_cann_ನಕ್ಷೆ) {
        GGML_ASSERT(!cann_ctx->ನಕ್ಷೆ_lru_cache.cache_list.ಖಾಲಿಗಳು());
        ggml_cann_ನಕ್ಷೆ * matched_ನಕ್ಷೆ = cann_ctx->ನಕ್ಷೆ_lru_cache.cache_list.front();

        ಆದರೆ (cann_ನಕ್ಷೆ_capture_required) {  // End CANN graph capture
            ACL_CHECK(aclmdl_RICapture_ಕೊನೆ(cann_ctx->stream(), &matched_ನಕ್ಷೆ->graph));
        }

        // Execute CANN graph
        ACL_CHECK(aclmdlRIExecuteAsync(matched_ನಕ್ಷೆ->graph, cann_ctx->stream()));
    }
#ಆದರೆ_ಅಂತ್ಯ  // USE_ACL_GRAPH
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
ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_cann_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_ಗಣ_device(cann_ctx->device);
    g_nz_workspaces[cann_ctx->device].clear();

    // calculate rope cache for fist layer in current device.
    cann_ctx->rope_cache.cached = ಸುಳ್ಳು;

    ತಾರ್ಕಿಕ ನಕ್ಷೆ_capture_required = ಸುಳ್ಳು;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ USE_ACL_ನಕ್ಷೆ
    ತಾರ್ಕಿಕ use_cann_ನಕ್ಷೆ = ನಿಜ;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ prefill_use_ನಕ್ಷೆ = parse_bool(get_env_as_lowercase("GGML_CANN_PREFILL_USE_GRAPH").value_or(""));
    ಆದರೆ (!prefill_use_ನಕ್ಷೆ) {
        // Do not use acl_graph for prefill.
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * ಗಂಟು = cgraph->nodes[i];
            // TODO: Optimize here. Currently, we can only
            // get seq_len by FA's input.
            ಆದರೆ (ಗಂಟು->op == GGML_OP_FLASH_ATTN_EXT) {
                // Q -> src[0], shape: [B, S, N, D]
                use_cann_ನಕ್ಷೆ = (ಗಂಟು->src[0]->ne[1] == 1);
                ನಿಲ್ಲಿಸು;
            }
        }
    }

    ಆದರೆ (!cann_ctx->acl_ನಕ್ಷೆ_ಬಹುಲಕ) {
        use_cann_ನಕ್ಷೆ = ಸುಳ್ಳು;
    }

    ಆದರೆ (use_cann_ನಕ್ಷೆ) {
        // If no matching graph is found, the graph needs to be recaptured.
        ನಕ್ಷೆ_capture_required = !cann_ctx->ನಕ್ಷೆ_lru_cache.ಕಂಡುಹಿಡಿ_and_move_ಗಮ್ಯ_front(cgraph);

        ಆದರೆ (ನಕ್ಷೆ_capture_required) {
            // If no matching graph is found, add a new ACL graph.
            ggml_cann_ನಕ್ಷೆ * new_ನಕ್ಷೆ = ggml_cann_ನಕ್ಷೆ::create_from_cgraph(cgraph);
            cann_ctx->ನಕ್ಷೆ_lru_cache.ತಳ್ಳು(new_ನಕ್ಷೆ);

            // Pre-load rope cache before graph capture.  During capture the
            // stream cannot perform host-to-device memcpy or device memory
            // malloc/free.  Running the full cache init now populates the
            // cache metadata so these branches are skipped during capture,
            // while also warming up the memory pool.
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * ಗಂಟು = cgraph->nodes[i];
                ಆದರೆ (ಗಂಟು->op == GGML_OP_ROPE) {
                    ggml_cann_rope_cache_preload(*cann_ctx, ಗಂಟು);
                    ನಿಲ್ಲಿಸು;
                }
            }
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ತಾರ್ಕಿಕ use_cann_ನಕ್ಷೆ = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ  // USE_ACL_GRAPH
    evaluate_and_capture_cann_ನಕ್ಷೆ(cann_ctx, cgraph, use_cann_ನಕ್ಷೆ, ನಕ್ಷೆ_capture_required);

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cann_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(op)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(op)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            {
                ಆಯ್ಕೆ (op->src[0]->type) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ASCEND_310P
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
#ಆದರೆ_ಅಂತ್ಯ
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
                        // Q4 && Q8 per group is not support on 310p device
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಇಲ್ಲದಿದ್ದರೆ
                        // only support contiguous for quantized types.
                        ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#ಆದರೆ_ಅಂತ್ಯ
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            ಆಯ್ಕೆ (op->src[0]->type) {
                ಸ್ಥಿತಿ GGML_TYPE_F16:
                ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
                    // Q4 && Q8 per group is not support on 310p device
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಇಲ್ಲದಿದ್ದರೆ
                    // only support contiguous for quantized types.
                    ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#ಆದರೆ_ಅಂತ್ಯ
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        // embedding
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ASCEND_310P
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
#ಆದರೆ_ಅಂತ್ಯ
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ಆಯ್ಕೆ (op->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ASCEND_310P
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
#ಆದರೆ_ಅಂತ್ಯ
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CPY:
            {
                ggml_tensor * src = op->src[0];
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
                ಆದರೆ ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                    // only support F32 and F16 on 310P.
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
#ಇಲ್ಲದಿದ್ದರೆ
                ಆದರೆ ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
                    // only support F32, F16 and BF16.
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
#ಆದರೆ_ಅಂತ್ಯ
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONT:
            {
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ASCEND_310P
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
#ಆದರೆ_ಅಂತ್ಯ
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            {
                ಆದರೆ (((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[15] != 0) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; // FIXME: support ggml_rope_set_offset
                }
                ಆದರೆ (op->src[0]->ne[0] > 896) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
                // TODO: Support rope_dim < ne00(dim)
                ಆದರೆ (op->src[0]->ne[0] != op->op_params[1]) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (!ggml_is_contiguous(op->src[0])) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
#ಆದರೆ_ಅಂತ್ಯ
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
            {
                // aclnnUpsampleNearest2dGetWorkspaceSize not support
                // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
                ಆದರೆ (op->src[0]->ne[2] * op->ne[3] != op->src[0]->ne[3] * op->ne[2]) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->op_params[0] != GGML_SCALE_ಬಹುಲಕ_ಹತ್ತಿರದ) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_POOL_2D:
            {
                ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *) op->op_params;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
                ಎಣಿಕೆ ggml_op_pool opt = static_cast<ggml_op_pool>(opts[0]);
                ಆದರೆ (opt == GGML_OP_POOL_MAX) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
#ಆದರೆ_ಅಂತ್ಯ
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k0 = opts[1];
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k1 = opts[2];
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p0 = opts[5];
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ p1 = opts[6];
                // value of paddingH should be at most half of kernelH
                // value of paddingW should be at most half of kernelW
                ಹಿಂತಿರುಗಿ (p0 <= (k0 / 2)) && (p1 <= (k1 / 2));
            }
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0]);
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
        ಸ್ಥಿತಿ GGML_OP_DUP:
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_NORM:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_ADD1:
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_DIV:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ಸ್ಥಿತಿ GGML_OP_SQR:
        ಸ್ಥಿತಿ GGML_OP_SQRT:
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
        ಸ್ಥಿತಿ GGML_OP_ACC:
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
        ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_PAD:
            // TODO: add circular padding support for cann, see https://github.com/ggml-org/llama.cpp/pull/16985
            ಹಿಂತಿರುಗಿ ggml_get_op_params_i32(op, 8) == 0;
        ಸ್ಥಿತಿ GGML_OP_ARANGE:
        ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
        ಸ್ಥಿತಿ GGML_OP_COS:
        ಸ್ಥಿತಿ GGML_OP_SIN:
        ಸ್ಥಿತಿ GGML_OP_LOG:
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
        ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
        ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
            {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
                // Ger is not supported on 310p device
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಇಲ್ಲದಿದ್ದರೆ
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            ದಶಮಾಂಶ bias;
            ಸ್ಮರಣೆ_ನಕಲು(&bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (op->op_params) + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
            ಹಿಂತಿರುಗಿ bias == 0.0f;  // TODO: support bias != 0.0f
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            // TODO: support attention sinks [TAG_ATTN_SINKS]
            ಆದರೆ (op->src[2]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ASCEND_310P
                // FA not support on 310p device
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಇಲ್ಲದಿದ್ದರೆ
                // derived from [ggml-cuda.cu]
                ಆದರೆ (op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32 &&
                    op->src[1]->type != GGML_TYPE_BF16) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                // TODO: support attention sinks [TAG_ATTN_SINKS]
                ಆದರೆ (op->src[4]) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                    // different head sizes of K and V are not supported yet
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ದಶಮಾಂಶ logitSoftcap = 0.0f;
                ಸ್ಮರಣೆ_ನಕಲು(&logitSoftcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (op->op_params) + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಆದರೆ (logitSoftcap != 0.0f) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
            }
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_TRI:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_FILL:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_DIAG:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    GGML_ಬಳಸದ(dev);
}

/**
 * @brief Records an event on the CANN backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_event_ದಾಖಲೆ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrt_ದಾಖಲೆ_Event((aclrtEvent) event->context, cann_ctx->stream()));
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ಆದರೆ (ggml_backend_is_cann(backend)) {
        ACL_CHECK(aclrt_ಹರಿವು_Wait_Event(cann_ctx->stream(), (aclrtEvent) event->context));
    } ಇಲ್ಲದಿದ್ದರೆ {
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
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_cann_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ggml_backend_cann_event_ದಾಖಲೆ,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .graph_optimize          = */ ಖಾಲಿ,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
ಸ್ಥಿರ ggml_guid_t ggml_backend_cann_guid() {
    ಸ್ಥಿರ ggml_guid guid = { 0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                              0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64 };
    ಹಿಂತಿರುಗಿ &guid;
}

// backend device
ರಚನೆ ggml_backend_cann_device_context {
    ಪೂರ್ಣಾಂಕ         device;
    std::string name;
    std::string description;
    ಪೂರ್ಣಾಂಕ op_offload_min_batch_ಗಾತ್ರ;
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cann_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ggml_backend_cann_get_device_ಸ್ಮರಣೆ(ctx->device, ಬಿಡುಗಡೆ, ಒಟ್ಟು);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_cann_device_get_type(ggml_backend_dev_t dev) {
    GGML_ಬಳಸದ(dev);
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type        = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);

    ತಾರ್ಕಿಕ host_ಬಫರ್ = getenv("GGML_CANN_NO_PINNED") == ಶೂನ್ಯಸೂಚಕ;

    props->caps = {
        /* .async                 = */ ಸುಳ್ಳು,
        /* .host_buffer           = */ host_ಬಫರ್,
        /* .buffer_from_host_ptr  = */ ಸುಳ್ಳು,
        /* .events                = */ ನಿಜ,
        /* .mmap_support          = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    GGML_ಬಳಸದ(params);
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ ggml_backend_cann_init(ctx->device);
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cann_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಆದರೆ (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context *      dev_ctx  = (ggml_backend_cann_device_context *) dev->context;
        ggml_backend_cann_ಬಫರ್_type_context * buft_ctx = (ggml_backend_cann_ಬಫರ್_type_context *) buft->context;
        ಹಿಂತಿರುಗಿ buft_ctx->device == dev_ctx->device;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_cann_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ ggml_backend_cann_ಬಫರ್_type(ctx->device);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_cann_device_get_host_ಬಫರ್_type(ggml_backend_dev_t dev) {
    GGML_ಬಳಸದ(dev);
    ಹಿಂತಿರುಗಿ ggml_backend_cann_host_ಬಫರ್_type();
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
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cann_offload_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;

    ಹಿಂತಿರುಗಿ op->ne[1] >= dev_ctx->op_offload_min_batch_ಗಾತ್ರ && op->op != GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು;
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
ಸ್ಥಿರ ggml_backend_event_t ggml_backend_cann_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *) dev->context;

    ggml_cann_ಗಣ_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_backend_event{
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent) event->context));

    ಅಳಿಸು event;
    GGML_ಬಳಸದ(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cann_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent) event->context));

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_ಸ್ಮರಣೆ,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,  // called for every card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type    = */ ggml_backend_cann_device_get_host_ಬಫರ್_type,
    /* .buffer_from_host_ptr    = */ ಖಾಲಿ,  // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};

// backend reg
ರಚನೆ ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_ಬಳಸದ(reg);
    ಹಿಂತಿರುಗಿ GGML_CANN_NAME;
}

ಸ್ಥಿರ size_t ggml_backend_cann_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    ಹಿಂತಿರುಗಿ ctx->devices.size();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < ctx->devices.size());
    ಹಿಂತಿರುಗಿ ctx->devices[ಸೂಚ್ಯಂಕ];
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_cann_reg_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    GGML_ಬಳಸದ(reg);
    GGML_ಬಳಸದ(name);
    // reserved for future use
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_ಎಣಿಕೆ,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_ವಿಳಾಸ,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    ಸ್ಥಿರ ggml_backend_reg reg;
    ಸ್ಥಿರ ತಾರ್ಕಿಕ             initialized = ಸುಳ್ಳು;

    {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);
        ಆದರೆ (!initialized) {
            aclInit(ಶೂನ್ಯಸೂಚಕ);
            ggml_backend_cann_reg_context * ctx = ಹೊಸ ggml_backend_cann_reg_context;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ min_batch_ಗಾತ್ರ = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_cann_info().device_ಎಣಿಕೆ; i++) {
                ggml_backend_cann_device_context * dev_ctx = ಹೊಸ ggml_backend_cann_device_context();
                dev_ctx->description                       = aclrtGetSocName();
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_CANN_NAME + std::to_string(i);
                dev_ctx->op_offload_min_batch_ಗಾತ್ರ         = min_batch_ಗಾತ್ರ;
                ggml_cann_ಗಣ_device(i);
                ggml_backend_dev_t dev = ಹೊಸ ggml_backend_device{ /* .iface   = */ ggml_backend_cann_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
                                    /* .iface       = */ ggml_backend_cann_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(ಶೂನ್ಯಸೂಚಕ);
    ಆದರೆ (device < 0 || device >= ggml_backend_cann_get_device_ಎಣಿಕೆ()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ggml_backend_cann_context * ctx = ಹೊಸ ggml_backend_cann_context(device);
    ಆದರೆ (ctx == ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ggml_cann_ಗಣ_device(ctx->device);
    ggml_backend_t cann_backend =
        ಹೊಸ ggml_backend{ /* .guid      = */ ggml_backend_cann_guid(),
                          /* .interface = */ ggml_backend_cann_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                          /* .context   = */ ctx };

    ಹಿಂತಿರುಗಿ cann_backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_cann(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_ಎಣಿಕೆ() {
    ಹಿಂತಿರುಗಿ ggml_cann_info().device_ಎಣಿಕೆ;
}

ಶೂನ್ಯ ggml_backend_cann_get_device_description(int32_t device, ಅಕ್ಷರ * description, size_t description_ಗಾತ್ರ) {
    ggml_cann_ಗಣ_device(device);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * soc_name = aclrtGetSocName();
    snprintf(description, description_ಗಾತ್ರ, "%s", soc_name);
}

ಶೂನ್ಯ ggml_backend_cann_get_device_ಸ್ಮರಣೆ(int32_t device, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ggml_cann_ಗಣ_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, ಬಿಡುಗಡೆ, ಒಟ್ಟು));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)

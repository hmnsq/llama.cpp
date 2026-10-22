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

#ضمن "ggml-cann.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cann/aclnn_ops.h"
#ضمن "ggml-cann/common.h"
#ضمن "ggml-impl.h"
#ضمن "ggml.h"

#ضمن <acl/acl.h>
#ضمن <aclnnop/aclnn_trans_matmul_weight.h>
#ضمن <stdarg.h>

#ضمن <chrono>
#ضمن <cmath>
#ضمن <cstdio>
#ضمن <cstring>
#ضمن <memory>
#ضمن <mutex>
#ضمن <optional>
#ضمن <queue>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <vector>

#تعریف_کرو GGML_COMMON_DECL_C

#ضمن "ggml-common.h"

#تعریف_کرو GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] فراغ ggml_cann_error(ثابت حرف * stmt, ثابت حرف * func, ثابت حرف * file, صحيح سطر, ثابت حرف * پیغام) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", پیغام);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, سطر);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

// Thread-local variable to record the current device of this thread.
thread_local صحيح g_current_cann_device = -1;

/**
 * @brief Set the CANN device to be used.
 *
 * @param device The target device ID to set.
 */
فراغ ggml_cann_سیٹ_device(ثابت int32_t device) {
    // int current_device = -1;
    // Note: In some CANN versions, if no device has been set yet,
    //       aclrtGetDevice(&current_device) may return 0 by default.
    // aclrtGetDevice(&current_device);

    // If the current device is already the target one, no need to switch.
    إذا (device == g_current_cann_device) {
        أرجع;
    }

    // Switch to the new device.
    ACL_CHECK(aclrt_سیٹ_Device(device));

    // Update the global device record.
    g_current_cann_device = device;
}

/**
 * @brief Get the value of the specified environment variable (name) as lowercase.
 *        if not empty, return a std::string object
 */
std::optional<std::string> get_env_as_lowercase(ثابت std::string & name) {
    ثابت حرف * val = std::getenv(name.c_str());
    إذا (!val) {
        أرجع std::nullopt;
    }
    std::string res = std::string(val);
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    أرجع res;
}

/**
 * @brief Verify whether the environment variable is a valid value.
 */
منطقي parse_bool(ثابت std::string & value) {
    ساكن ثابت std::unordered_set<std::string> valid_اقدار = { "on", "1", "yes", "y", "enable", "true" };
    أرجع valid_اقدار.find(value) != valid_اقدار.end();
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
صحيح parse_integer(ثابت std::string & value) {
    حاول {
        أرجع std::stoi(value);
    } امسك (...) {
        أرجع 0;
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
ساكن ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrt_Get_Device_گنتی((uint32_t *) &info.device_گنتی);

    إذا (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n", __func__, acl_Get_Recent_Err_پیغام());
        أرجع info;
    }

    GGML_ASSERT(info.device_گنتی <= GGML_CANN_MAX_DEVICES);

    لكل (صحيح id = 0; id < info.device_گنتی; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
        prop.ونڈ_Type       = ACL_MEM_ونڈ_TYPE_PINNED;
        prop.memAttr              = ACL_HBM_MEM_HUGE;
        prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = id;
        prop.reserve              = 0;
        err                       = aclrt_Mem_Get_ونڈ_Granularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                                                                     &info.devices[id].vmm_granularity);
        info.devices[id].vmm      = err == ACL_SUCCESS;

        size_t حرر, کل;
        ggml_backend_cann_get_device_حافظہ(id, &حرر, &کل);
        info.devices[id].کل_vram = حرر;
    }

    // TODO: add more device info later.
    أرجع info;
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
ثابت ggml_cann_device_info & ggml_cann_info() {
    ساكن ggml_cann_device_info info = ggml_cann_init();
    أرجع info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(priority segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
بنية ggml_cann_pool_buf_prio : عام ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    ساكن ثابت size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    ساكن ثابت size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    ساكن ثابت size_t alignment = 128;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    صحيح device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    منطقي disable_clean = خطأ;

    /**
     * @brief Structure representing a CANN buffer.
     */
    بنية ggml_cann_بفر {
        فراغ *                                ptr  = خالی_سوچک;  ///< Pointer to the buffer.
        size_t                                size = 0;        ///< Size of the buffer.
        std::chrono::steady_clock::time_point آخری_ورتیا;       ///< Last used time.

        منطقي عامل>(ثابت ggml_cann_بفر & other) ثابت { أرجع size > other.size; }
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    std::unordered_map<فراغ *, size_t>                                                   بفر_pool;
    std::priority_queue<ggml_cann_بفر, std::vector<ggml_cann_بفر>, std::greater<>> free_buffers;

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_ناپ = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    واضح ggml_cann_pool_buf_prio(صحيح device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf_prio() {
        ggml_cann_سیٹ_device(device);
        لكل (خودکار & [b_ptr, b_ناپ] : بفر_pool) {
            aclrtFree(b_ptr);
            pool_ناپ -= b_ناپ;
        }
        بفر_pool.clear();
        GGML_ASSERT(pool_ناپ == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    فراغ * alloc(size_t size, size_t * actual_ناپ) override {
        size = GGML_PAD(size, alignment);
        إذا (size == 0) {
            size = alignment;
        }

        فراغ * ptr = خالی_سوچک;
        خودکار   now = std::chrono::steady_clock::now();

        std::vector<ggml_cann_بفر> free_buffers_rest;
        free_buffers_rest.reserve(free_buffers.size());
        طالما (!free_buffers.خالی_تھاواں()) {
            خودکار b = free_buffers.top();
            free_buffers.نکالو();

            إذا (b.size >= size) {
                // reuse the buffer if the size is enough
                ثابت size_t margin = b.size - size;
                إذا (margin <= max_reuse_margin) {
                    *actual_ناپ = b.size;
                    ptr          = b.ptr;
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#جے_انت
                    اكسر;
                }
            }

            منطقي should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.آخری_ورتیا).count() > 100;
            إذا (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_ناپ -= b.size;
                بفر_pool.erase(b.ptr);
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#جے_انت
                استمر;
            }
            free_buffers_rest.push_back(b);
        }
        لكل (ggml_cann_بفر & b : free_buffers_rest) {
            free_buffers.ڈالو(std::move(b));
        }

#جے_تعریف_شدہ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d] free pool_size = %5u MB\n\n", device,
                      (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576));
#جے_انت
        إذا (ptr != خالی_سوچک) {
            أرجع ptr;
        }

        // allocate a new buffer if no buffer can be reused
        ggml_cann_سیٹ_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_پہلا));
        *actual_ناپ = size;
        pool_ناپ += size;
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: allocate %p, "
            "pool_size = %5u MB, "
            "size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576),
            (uint32_t) (GGML_PAD(size, 1048576) / 1048576));
#جے_انت
        بفر_pool.emplace(ptr, size);
        أرجع ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    فراغ حرر(فراغ * ptr, size_t size) override {
        GGML_بن_ورتیا(size);
        خودکار it = بفر_pool.find(ptr);
        إذا (it == بفر_pool.end()) {
            GGML_ABORT("cann pool[%d]: buffer %p not found in pool\n", device, ptr);
        }

        خودکار now = std::chrono::steady_clock::now();
        free_buffers.emplace(ggml_cann_بفر{ ptr, it->second, now });
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: return   %p, "
            "pool_size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576));
#جے_انت
    }
};

/**
 * @brief A pool of CANN buffers(segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
بنية ggml_cann_pool_buf : عام ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    ساكن ثابت size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    ساكن ثابت size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    ساكن ثابت size_t alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    ساكن ثابت صحيح MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    صحيح device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    منطقي disable_clean = خطأ;

    /**
     * @brief Structure representing a CANN buffer.
     */
    بنية ggml_cann_بفر {
        فراغ *                                ptr  = خالی_سوچک;  ///< Pointer to the buffer memory.
        size_t                                size = 0;        ///< Size of the buffer.
        منطقي                                  ورتیا = خطأ;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point آخری_ورتیا;       ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_بفر بفر_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_ناپ = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    واضح ggml_cann_pool_buf(صحيح device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf() {
        ggml_cann_سیٹ_device(device);
        لكل (صحيح i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_بفر & b = بفر_pool[i];
            إذا (b.ptr != خالی_سوچک) {
                aclrtFree(b.ptr);
                pool_ناپ -= b.size;
            }
        }
        GGML_ASSERT(pool_ناپ == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    فراغ * alloc(size_t size, size_t * actual_ناپ) override {
        size = GGML_PAD(size, alignment);
        إذا (size == 0) {
            size = alignment;
        }

        فراغ * ptr = خالی_سوچک;
        خودکار   now = std::chrono::steady_clock::now();

        صحيح i = 0;
        لكل (; i < MAX_BUFFERS; ++i) {
            ggml_cann_بفر & b = بفر_pool[i];
            إذا (b.ptr == خالی_سوچک) {
                اكسر;
            }
            إذا (b.ورتیا) {
                استمر;
            }
            إذا (b.size >= size) {
                // reuse the buffer if the size is enough
                ثابت size_t margin = b.size - size;
                إذا (margin <= max_reuse_margin) {
                    *actual_ناپ = b.size;
                    b.ورتیا       = سچ;
                    ptr          = b.ptr;
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#جے_انت
                    اكسر;
                }
            }

            منطقي should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.آخری_ورتیا).count() > 100;
            إذا (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_ناپ -= b.size;
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#جے_انت
                b.ptr = خالی_سوچک;
            }
        }
        إذا (ptr != خالی_سوچک) {
            أرجع ptr;
        }

        إذا (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_cann_بفر & b = بفر_pool[i];
            ggml_cann_سیٹ_device(device);
            ACL_CHECK(aclrtMalloc(&b.ptr, size, ACL_MEM_MALLOC_HUGE_پہلا));
            pool_ناپ += size;
            *actual_ناپ = size;
            b.size       = size;
            b.ورتیا       = سچ;
            إذا (i >= MAX_BUFFERS - 8) {
                GGML_LOG_WARN("cann pool[%d]: slots almost full\n", device);
            }
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576),
                (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#جے_انت
            أرجع b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    فراغ حرر(فراغ * ptr, size_t size) override {
        GGML_بن_ورتیا(size);
        لكل (صحيح i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_بفر & b = بفر_pool[i];
            إذا (b.ptr != ptr) {
                استمر;
            }
            b.ورتیا      = خطأ;
            b.آخری_ورتیا = std::chrono::steady_clock::now();
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_ناپ, 1048576) / 1048576));
#جے_انت
            أرجع;
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
بنية ggml_cann_pool_vmm : عام ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    size_t max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    صحيح device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    فراغ * pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_ورتیا = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_ناپ = 0;

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
    std::vector<فراغ *> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    واضح ggml_cann_pool_vmm(صحيح device) : device(device) {
        خودکار dev    = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size    = dev.کل_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        إذا (pool_addr != 0) {
            لكل (خودکار & وتھ : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(وتھ));
            }
            لكل (خودکار & handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrt_Release_Mem_پتہ(pool_addr));
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
    فراغ * alloc(size_t size, size_t * actual_ناپ) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        ثابت size_t alignment = 128;
        size                   = GGML_PAD(size, alignment);
        إذا (size == 0) {
            size = alignment;
        }

        size_t avail = pool_ناپ - pool_ورتیا;

        إذا (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_ناپ = size - avail;
            reserve_ناپ        = GGML_PAD(reserve_ناپ, granularity);

            GGML_ASSERT(pool_ناپ + reserve_ناپ <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
            prop.ونڈ_Type       = ACL_MEM_ونڈ_TYPE_PINNED;
            prop.memAttr              = ACL_HBM_MEM_HUGE;
            prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id          = device;
            prop.reserve              = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_ناپ, &prop, 0));

            // reserve virtual address space (if not already reserved)
            إذا (pool_addr == 0) {
                ACL_CHECK(aclrt_Reserve_Mem_پتہ(&pool_addr, max_size, 0, عدم, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((حرف *) pool_addr + pool_ناپ, reserve_ناپ, 0, handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((حرف *) pool_addr + pool_ناپ);

            // add to the pool
            pool_ناپ += reserve_ناپ;

#جے_تعریف_شدہ DEBUG_CANN_MALLOC
            GGML_LOG_INFO("cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n", device,
                          (بنانشانی لما لما) (pool_ناپ / 1024 / 1024),
                          (بنانشانی لما لما) (reserve_ناپ / 1024 / 1024));
#جے_انت
        }

        GGML_ASSERT(pool_addr != 0);

        فراغ * ptr   = (فراغ *) ((حرف *) pool_addr + pool_ورتیا);
        *actual_ناپ = size;
        pool_ورتیا += size;

#جے_تعریف_شدہ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device, (بنانشانی لما لما) size,
                      (بنانشانی لما لما) ptr);
#جے_انت
        أرجع ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    فراغ حرر(فراغ * ptr, size_t size) override {
#جے_تعریف_شدہ DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device, (بنانشانی لما لما) size,
                      (بنانشانی لما لما) ptr);
#جے_انت

        pool_ورتیا -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (فراغ *) ((حرف *) pool_addr + pool_ورتیا));
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
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(صحيح device) {
    std::string mem_pool_type = get_env_as_lowercase("GGML_CANN_MEM_POOL").value_or("");

    إذا (mem_pool_type == "prio") {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        أرجع std::unique_ptr<ggml_cann_pool>(جديد ggml_cann_pool_buf_prio(device));
    }

    إذا (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        أرجع std::unique_ptr<ggml_cann_pool>(جديد ggml_cann_pool_vmm(device));
    }

    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    أرجع std::unique_ptr<ggml_cann_pool>(جديد ggml_cann_pool_buf(device));
}

// cann buffer

/**
 * @brief Tracks multi-threaded write progress for a single tensor.
 *
 * When multiple threads call set_tensor on different chunks of the same tensor,
 * this tracker accumulates progress and defers post-processing (quantized format
 * transform or ND-to-NZ conversion) until all data has been written.
 */
بنية Tensor_سیٹ_Tracker {
    std::mutex mtx;                   ///< Protects concurrent access to this tracker
    size_t بائٹاں_written = 0;         ///< Accumulated bytes written so far
    size_t کل_بائٹاں = 0;           ///< Target size (full tensor)
    std::vector<uint8_t> host_بفر; ///< Host staging buffer for quantized tensors
};

/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
بنية ggml_backend_cann_بفر_context {
    int32_t device;             ///< The device ID associated with this buffer context.
    فراغ *  dev_ptr = خالی_سوچک;  ///< Pointer to the device memory allocated for the buffer.

    std::mutex tracker_mutex;   ///< Protects the trackers map
    std::unordered_map<فراغ *, std::unique_ptr<Tensor_سیٹ_Tracker>> trackers;

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_بفر_context(int32_t device, فراغ * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_بفر_context() { ACL_CHECK(aclrtFree(dev_ptr)); }

    /**
     * @brief Get or create a tracker for the given tensor.
     */
    Tensor_سیٹ_Tracker * get_or_create_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> جندرا(tracker_mutex);
        خودکار کلید = tensor->انکڑے;
        خودکار it = trackers.find(کلید);
        إذا (it == trackers.end()) {
            خودکار tracker = std::make_unique<Tensor_سیٹ_Tracker>();
            tracker->کل_بائٹاں = ggml_nbytes(tensor);
            خودکار * ptr = tracker.get();
            trackers[کلید] = std::move(tracker);
            أرجع ptr;
        }
        أرجع it->second.get();
    }

    /**
     * @brief Remove the tracker for the given tensor.
     */
    فراغ remove_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> جندرا(tracker_mutex);
        trackers.erase(tensor->انکڑے);
    }
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
بنية ggml_backend_cann_بفر_type_context {
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
ساكن ثابت حرف * ggml_backend_cann_بفر_type_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_cann_بفر_type_context * buft_ctx = (ggml_backend_cann_بفر_type_context *) buft->context;

    أرجع buft_ctx->name.c_str();
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
ساكن منطقي ggml_backend_buft_is_cann(ggml_backend_بفر_type_t buft) {
    أرجع buft->iface.get_name == ggml_backend_cann_بفر_type_name;
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
ساكن فراغ ggml_backend_cann_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_cann_بفر_context * ctx = (ggml_backend_cann_بفر_context *) بفر->context;
    مٹاؤ ctx;
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
ساكن فراغ * ggml_backend_cann_بفر_get_ادھار(ggml_backend_بفر_t بفر) {
    ggml_backend_cann_بفر_context * ctx = (ggml_backend_cann_بفر_context *) بفر->context;
    أرجع ctx->dev_ptr;
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
ساكن فراغ ggml_backend_cann_transform_q4_0(ggml_tensor * tensor, ثابت فراغ * src, فراغ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_بائٹاں = n_elems * حجم(uint8_t) / 2;

    uint8_t *  quant_وتھ = (uint8_t *) dst;
    uint16_t * scale_وتھ = (uint16_t *) ((حرف *) dst + quant_بائٹاں);

    لكل (صحيح i = 0; i < groups; i++) {
        ثابت بلاک_q4_0 * گروہ = (ثابت بلاک_q4_0 *) ((ثابت حرف *) src + i * حجم(بلاک_q4_0));
        *scale_وتھ            = گروہ->d;
        scale_وتھ++;

        // 0-15
        لكل (صحيح j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_وتھ) = (گروہ->qs[j] & 0x0F);
            (*quant_وتھ) |= ((گروہ->qs[j + 1] << 4));
            quant_وتھ++;
        }

        // 16-31
        لكل (صحيح j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_وتھ) = (گروہ->qs[j] >> 4);
            (*quant_وتھ) |= (گروہ->qs[j + 1] & 0xF0);
            quant_وتھ++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    لكل (quant_وتھ = (uint8_t *) dst; quant_وتھ < (uint8_t *) dst + quant_بائٹاں; quant_وتھ++) {
        (*quant_وتھ) ^= 0x88;
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
ساكن فراغ ggml_backend_cann_transform_پٹھ_q4_0(ثابت ggml_tensor * tensor, فراغ * src, فراغ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_بائٹاں = n_elems * حجم(uint8_t) / 2;

    uint8_t *  quant_وتھ = (uint8_t *) src;
    uint16_t * scale_وتھ = (uint16_t *) ((حرف *) src + quant_بائٹاں);

    لكل (; quant_وتھ < (uint8_t *) src + quant_بائٹاں; quant_وتھ++) {
        (*quant_وتھ) ^= 0x88;
    }
    quant_وتھ = (uint8_t *) src;

    لكل (صحيح i = 0; i < groups; i++) {
        بلاک_q4_0 * گروہ = (بلاک_q4_0 *) ((حرف *) dst + i * حجم(بلاک_q4_0));
        گروہ->d           = *scale_وتھ;
        scale_وتھ++;

        // 0-15
        لكل (صحيح j = 0; j < QK4_0 / 2; j += 2) {
            گروہ->qs[j]     = ((*quant_وتھ) & 0x0F);
            گروہ->qs[j + 1] = ((*quant_وتھ) >> 4);
            quant_وتھ++;
        }

        // 16-31
        لكل (صحيح j = 0; j < QK4_0 / 2; j += 2) {
            گروہ->qs[j] |= ((*quant_وتھ) << 4);
            گروہ->qs[j + 1] |= ((*quant_وتھ) & 0xF0);
            quant_وتھ++;
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
ساكن فراغ ggml_backend_cann_transform_q8_0(ggml_tensor * tensor, ثابت فراغ * src, فراغ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_بائٹاں = n_elems * حجم(uint8_t);

    uint8_t *  quant_وتھ = (uint8_t *) dst;
    uint16_t * scale_وتھ = (uint16_t *) ((حرف *) dst + quant_بائٹاں);

    لكل (صحيح i = 0; i < groups; i++) {
        ثابت بلاک_q8_0 * گروہ = (ثابت بلاک_q8_0 *) ((ثابت حرف *) src + i * حجم(بلاک_q8_0));
        *scale_وتھ            = گروہ->d;
        scale_وتھ++;
        size_t گروہ_quant_ناپ = QK8_0 * حجم(uint8_t);
        حافظہ_نقل(quant_وتھ, گروہ->qs, گروہ_quant_ناپ);
        quant_وتھ += گروہ_quant_ناپ;
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
ساكن فراغ ggml_backend_cann_transform_پٹھ_q8_0(ثابت ggml_tensor * tensor, ثابت فراغ * src, فراغ * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_بائٹاں = n_elems * حجم(uint8_t);

    ثابت uint8_t *  quant_وتھ = (ثابت uint8_t *) src;
    ثابت uint16_t * scale_وتھ = (ثابت uint16_t *) ((ثابت حرف *) src + quant_بائٹاں);

    لكل (صحيح i = 0; i < groups; i++) {
        بلاک_q8_0 * گروہ = (بلاک_q8_0 *) ((حرف *) dst + i * حجم(بلاک_q8_0));
        گروہ->d           = *scale_وتھ;
        scale_وتھ++;
        size_t گروہ_quant_ناپ = QK8_0 * حجم(uint8_t);
        حافظہ_نقل(گروہ->qs, quant_وتھ, گروہ_quant_ناپ);
        quant_وتھ += گروہ_quant_ناپ;
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
ساكن فراغ ggml_backend_cann_transform(ggml_tensor * tensor, ثابت فراغ * src, فراغ * dst) {
    بدل (tensor->type) {
        حالة GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            اكسر;
        افتراضي:
            اكسر;
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
ساكن فراغ ggml_backend_cann_transform_پٹھ(ثابت ggml_tensor * tensor, فراغ * src, فراغ * dst) {
    بدل (tensor->type) {
        حالة GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_پٹھ_q4_0(tensor, src, dst);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_پٹھ_q8_0(tensor, src, dst);
            اكسر;
        افتراضي:
            اكسر;
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
ساكن منطقي need_transform(ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q8_0:
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
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
ساكن تعداد ggml_حالت ggml_backend_cann_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    إذا (tensor->view_src != عدم && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->بفر->buft == بفر->buft);
        أرجع GGML_حالت_SUCCESS;
    }

    // TODO: cann backend doesn't support quantized yet. Just leave the code
    // here.
    إذا (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_ناپ = ggml_nbytes(tensor);
        size_t padded_ناپ   = ggml_backend_buft_get_alloc_ناپ(بفر->buft, tensor);

        إذا (padded_ناپ > original_ناپ && tensor->view_src == خالی_سوچک) {
            size_t memset_ناپ = padded_ناپ - original_ناپ;
            ACL_CHECK(aclrtMemset((حرف *) tensor->انکڑے + original_ناپ, memset_ناپ, 0, memset_ناپ));
        }
    }
    أرجع GGML_حالت_SUCCESS;
}

/**
 * @brief Workspace for caching NZ buffers per device.
 *
 * This struct manages a device buffer used in NZ computations. It supports
 * allocation, reallocation, and clearing of cached memory. The struct is
 * designed to be used with a global array, one per device.
 */
بنية ggml_cann_nz_workspace {
    std::mutex mtx;    // Protects ptr/allocated from concurrent access
    فراغ * ptr;        // Pointer to allocated device buffer
    size_t allocated;  // Size of currently allocated buffer in bytes

    /**
     * @brief Constructor. Initializes the workspace with no allocated memory.
     */
    ggml_cann_nz_workspace() : ptr(خالی_سوچک), allocated(0) {}

    /**
     * @brief Free cached memory and reset the workspace.
     *
     * If a buffer has been allocated, this function releases it using
     * aclrtFree and resets internal state.
     */
    فراغ clear() {
        إذا (ptr) {
            ACL_CHECK(aclrtFree(ptr));
            ptr       = خالی_سوچک;
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
    فراغ realloc(size_t new_ناپ) {
        إذا (new_ناپ > allocated) {
            clear();
            ACL_CHECK(aclrtMalloc(&ptr, new_ناپ, ACL_MEM_MALLOC_HUGE_پہلا));
            allocated = new_ناپ;
        }
    }

    /**
     * @brief Get the device buffer pointer.
     *
     * @return Pointer to the allocated buffer, or nullptr if not allocated.
     */
    فراغ * get() ثابت { أرجع ptr; }
};

/**
 * @brief Global array of NZ workspaces, one per device.
 */
ساكن ggml_cann_nz_workspace g_nz_workspaces[GGML_CANN_MAX_DEVICES];

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
ساكن فراغ بھار_format_منزل_nz(ggml_tensor * tensor, صحيح device) {
    acl_tensor_ptr بھار_Transposed = ggml_cann_create_tensor(tensor, tensor->ne, tensor->nb, 2, ACL_FORMAT_ND, 0);
    uint64_t       workspace_ناپ    = 0;
    aclOpExecutor * executor;

    // TransMatmulWeight
    ACL_CHECK(aclnn_Trans_Matmul_بھار_Get_Workspace_ناپ(بھار_Transposed.get(), &workspace_ناپ, &executor));

    std::lock_guard<std::mutex> جندرا(g_nz_workspaces[device].mtx);
    // Avoid frequent malloc/free of the workspace.
    g_nz_workspaces[device].realloc(workspace_ناپ);

    فراغ * g_nz_workspace = g_nz_workspaces[device].get();

    ACL_CHECK(aclnn_Trans_Matmul_بھار(g_nz_workspace, workspace_ناپ, executor, خالی_سوچک));
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
ساكن فراغ ggml_backend_cann_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                                ggml_tensor *         tensor,
                                                ثابت فراغ *          انکڑے,
                                                size_t                وتھ,
                                                size_t                size) {
    ggml_backend_cann_بفر_context * ctx = (ggml_backend_cann_بفر_context *) بفر->context;

    ggml_cann_سیٹ_device(ctx->device);

    // Only check env once.
    ساكن منطقي بھار_منزل_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    منطقي is_quantized = need_transform(tensor->type);
    منطقي is_nz        = !is_quantized && tensor->type != GGML_TYPE_BF16 && بھار_منزل_nz &&
                 is_matmul_بھار((ثابت ggml_tensor *) tensor);

    // Plain tensor (not quantized, not NZ): direct copy, no tracking needed
    إذا (!is_quantized && !is_nz) {
        ACL_CHECK(aclrtMemcpy((حرف *) tensor->انکڑے + وتھ, size, انکڑے, size, ACL_MEMCPY_HOST_منزل_DEVICE));
        أرجع;
    }

    // Single-shot write (full tensor at once): handle directly without tracking overhead
    إذا (وتھ == 0 && size == ggml_nbytes(tensor)) {
        إذا (is_quantized) {
            فراغ * transform_بفر = حافظہ_دیو(size);
            ggml_backend_cann_transform(tensor, انکڑے, transform_بفر);
            ACL_CHECK(aclrtMemcpy(tensor->انکڑے, size, transform_بفر, size, ACL_MEMCPY_HOST_منزل_DEVICE));
            حرر(transform_بفر);
        } وإلا {
            // NZ weight
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ACL_CHECK(aclrtMemcpy(tensor->انکڑے, size, انکڑے, size, ACL_MEMCPY_HOST_منزل_DEVICE));
            بھار_format_منزل_nz(tensor, ctx->device);
        }
        أرجع;
    }

    // Chunked write: use tracker to accumulate progress and defer transform/conversion
    Tensor_سیٹ_Tracker * tracker = ctx->get_or_create_tracker(tensor);
    std::unique_lock<std::mutex> جندرا(tracker->mtx);

    إذا (is_quantized) {
        // Stage data in host buffer; transform requires full tensor data
        إذا (tracker->host_بفر.خالی_تھاواں()) {
            tracker->host_بفر.resize(tracker->کل_بائٹاں);
        }
        حافظہ_نقل(tracker->host_بفر.انکڑے() + وتھ, انکڑے, size);
    } وإلا {
        // NZ weight: upload chunk to device immediately, defer conversion
        ACL_CHECK(aclrtMemcpy((حرف *) tensor->انکڑے + وتھ, size, انکڑے, size, ACL_MEMCPY_HOST_منزل_DEVICE));
    }

    tracker->بائٹاں_written += size;

    // All chunks received: perform deferred transform/conversion
    إذا (tracker->بائٹاں_written >= tracker->کل_بائٹاں) {
        إذا (is_quantized) {
            فراغ * transform_بفر = حافظہ_دیو(tracker->کل_بائٹاں);
            ggml_backend_cann_transform(tensor, tracker->host_بفر.انکڑے(), transform_بفر);
            ACL_CHECK(aclrtMemcpy(tensor->انکڑے, tracker->کل_بائٹاں, transform_بفر, tracker->کل_بائٹاں, ACL_MEMCPY_HOST_منزل_DEVICE));
            حرر(transform_بفر);
        }

        إذا (is_nz) {
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            بھار_format_منزل_nz(tensor, ctx->device);
        }

        // Unlock before removing tracker, as remove_tracker destroys the mutex
        جندرا.unlock();
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
ساكن فراغ ggml_backend_cann_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                ثابت ggml_tensor *   tensor,
                                                فراغ *                انکڑے,
                                                size_t                وتھ,
                                                size_t                size) {
    ggml_backend_cann_بفر_context * ctx = (ggml_backend_cann_بفر_context *) بفر->context;

    ggml_cann_سیٹ_device(ctx->device);

    إذا (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(انکڑے, size, (حرف *) tensor->انکڑے + وتھ, size, ACL_MEMCPY_DEVICE_منزل_HOST));
    } وإلا {
        فراغ * transform_بفر = حافظہ_دیو(size);
        ACL_CHECK(aclrtMemcpy(transform_بفر, size, (حرف *) tensor->انکڑے + وتھ, size, ACL_MEMCPY_DEVICE_منزل_HOST));
        ggml_backend_cann_transform_پٹھ(tensor, transform_بفر, انکڑے);
        حرر(transform_بفر);
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
ساكن منطقي ggml_backend_cann_بفر_cpy_tensor(ggml_backend_بفر_t بفر,
                                                ثابت ggml_tensor *   src,
                                                ggml_tensor *         dst) {
    إذا (ggml_backend_buft_is_cann(src->بفر->buft)) {
        ggml_backend_cann_بفر_context * src_ctx = (ggml_backend_cann_بفر_context *) src->بفر->context;
        ggml_backend_cann_بفر_context * dst_ctx = (ggml_backend_cann_بفر_context *) بفر->context;

        size_t memcpy_ناپ = ggml_nbytes(src);
        // Same device.
        إذا (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((حرف *) dst->انکڑے, memcpy_ناپ, (ثابت حرف *) src->انکڑے, memcpy_ناپ,
                                  ACL_MEMCPY_DEVICE_منزل_DEVICE));
            أرجع سچ;
        } وإلا {
#جے_تعریف_شدہ ASCEND_310P
            // TODO: Support 310p P2P copy
            أرجع خطأ;
#جے_انت
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device, dst_ctx->device));
            إذا (canAccessPeer) {
                ggml_cann_سیٹ_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((حرف *) dst->انکڑے, memcpy_ناپ, (ثابت حرف *) src->انکڑے, memcpy_ناپ,
                                      ACL_MEMCPY_DEVICE_منزل_DEVICE));
                أرجع سچ;
            }
        }
    }
    أرجع خطأ;
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
ساكن فراغ ggml_backend_cann_بفر_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t وتھ, size_t size) {
    ggml_backend_cann_بفر_context * ctx = (ggml_backend_cann_بفر_context *) بفر->context;

    ggml_cann_سیٹ_device(ctx->device);
    ACL_CHECK(aclrtMemset((حرف *) tensor->انکڑے + وتھ, size, value, size));
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
ساكن فراغ ggml_backend_cann_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_backend_cann_بفر_context * ctx = (ggml_backend_cann_بفر_context *) بفر->context;

    ggml_cann_سیٹ_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, بفر->size, value, بفر->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
ساكن ثابت ggml_backend_بفر_i ggml_backend_cann_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_cann_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_cann_بفر_get_ادھار,
    /* .init_tensor     = */ ggml_backend_cann_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cann_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cann_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_cann_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_cann_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_بفر_clear,
    /* .reset           = */ عدم,
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
ساكن ggml_backend_بفر_t ggml_backend_cann_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_cann_بفر_type_context * buft_ctx = (ggml_backend_cann_بفر_type_context *) buft->context;

    ggml_cann_سیٹ_device(buft_ctx->device);

    ثابت size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    إذا (size == 0) {
        size = alignment;
    }
    فراغ *   dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_پہلا);
    إذا (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, buft_ctx->device, acl_Get_Recent_Err_پیغام());
        أرجع خالی_سوچک;
    }

    ggml_backend_cann_بفر_context * ctx = جديد ggml_backend_cann_بفر_context(buft_ctx->device, dev_ptr);

    أرجع ggml_backend_بفر_init(buft, ggml_backend_cann_بفر_interface, ctx, size);
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
ساكن size_t ggml_backend_cann_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع 128;

    GGML_بن_ورتیا(buft);
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
ساكن size_t ggml_backend_cann_بفر_type_get_alloc_ناپ(ggml_backend_بفر_type_t buft,
                                                           ثابت ggml_tensor *        tensor) {
    size_t  size = ggml_nbytes(tensor);
    int64_t ne0  = tensor->ne[0];

    // Only check env once.
    ساكن منطقي بھار_منزل_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);
    إذا (ggml_is_quantized(tensor->type)) {
        إذا (ne0 % میٹرکس_صف_PADDING != 0) {
            size += ggml_صف_ناپ(tensor->type, میٹرکس_صف_PADDING - ne0 % میٹرکس_صف_PADDING);
        }
    } وإلا إذا (بھار_منزل_nz && tensor->type != GGML_TYPE_BF16
               && is_matmul_بھار((ثابت ggml_tensor *) tensor)) {
        // NZ format weight are not support quantized yet.
        // If ND tensor transform to NZ, size may changed.
        int64_t shape[] = { tensor->ne[1], tensor->ne[0] };
        GGML_ASSERT(tensor->ne[2] == 1);
        GGML_ASSERT(tensor->ne[3] == 1);
        ثابت aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
        size_t              new_ناپ;
        ACL_CHECK(aclnn_Calculate_Matmul_بھار_ناپ_V2(acl_shape, ggml_cann_type_mapping(tensor->type), &new_ناپ));
        ACL_CHECK(aclDestroyIntArray(acl_shape));
        size = std::max(size, new_ناپ);
    }

    أرجع size;

    GGML_بن_ورتیا(buft);
}

ساكن منطقي ggml_backend_cann_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    أرجع خطأ;

    GGML_بن_ورتیا(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ساكن ثابت ggml_backend_بفر_type_i ggml_backend_cann_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_cann_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_cann_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_cann_بفر_type_get_alignment,
    /* .get_max_size        = */ عدم,  // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cann_بفر_type_get_alloc_ناپ,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ ggml_backend_cann_بفر_type_is_host,
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
ggml_backend_بفر_type_t ggml_backend_cann_بفر_type(int32_t device) {
    ساكن std::mutex           mutex;
    std::lock_guard<std::mutex> جندرا(mutex);

    إذا (device >= ggml_backend_cann_get_device_گنتی()) {
        أرجع خالی_سوچک;
    }

    ساكن ggml_backend_بفر_type ggml_backend_cann_بفر_types[GGML_CANN_MAX_DEVICES];

    ساكن منطقي ggml_backend_cann_بفر_type_initialized = خطأ;

    إذا (!ggml_backend_cann_بفر_type_initialized) {
        لكل (int32_t i = 0; i < ggml_cann_info().device_گنتی; i++) {
            ggml_backend_cann_بفر_types[i] = {
                /* .iface    = */ ggml_backend_cann_بفر_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), i),
                /* .context  = */
                جديد ggml_backend_cann_بفر_type_context{ i, "CANN" + std::to_string(i) },
            };
        }
        ggml_backend_cann_بفر_type_initialized = سچ;
    }

    أرجع &ggml_backend_cann_بفر_types[device];
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
ساكن ثابت حرف * ggml_backend_cann_host_بفر_type_name(ggml_backend_بفر_type_t buft) {
    أرجع "CANN_Host";

    GGML_بن_ورتیا(buft);
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
ساكن ثابت حرف * ggml_backend_cann_host_بفر_name(ggml_backend_بفر_t بفر) {
    أرجع "CANN_Host";

    GGML_بن_ورتیا(بفر);
}

/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer, including
 * its context.
 *
 * @param buffer The CANN host buffer to free.
 */
ساكن فراغ ggml_backend_cann_host_بفر_free(ggml_backend_بفر_t بفر) {
    ACL_CHECK(aclrtFreeHost(بفر->context));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
ساكن فراغ * ggml_cann_host_malloc(size_t size) {
    إذا (getenv("GGML_CANN_NO_PINNED") != خالی_سوچک) {
        أرجع خالی_سوچک;
    }

    ثابت size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    إذا (size == 0) {
        size = alignment;
    }

    فراغ *   hostPtr = خالی_سوچک;
    aclError err     = aclrtMallocHost((فراغ **) &hostPtr, size);
    إذا (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                      acl_Get_Recent_Err_پیغام());
        أرجع خالی_سوچک;
    }
    أرجع hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
ساكن ggml_backend_بفر_t ggml_backend_cann_host_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                             size_t                     size) {
    فراغ * hostPtr = ggml_cann_host_malloc(size);

    إذا (hostPtr == خالی_سوچک) {
        // fallback to cpu buffer
        أرجع ggml_backend_buft_alloc_بفر(ggml_backend_cpu_بفر_type(), size);
    }

    ggml_backend_بفر_t بفر = ggml_backend_cpu_بفر_from_ptr(hostPtr, size);
    بفر->buft                 = buft;
    بفر->iface.free_بفر    = ggml_backend_cann_host_بفر_free;

    أرجع بفر;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_بفر_type_t ggml_backend_cann_host_بفر_type() {
    ساكن بنية ggml_backend_بفر_type ggml_backend_cann_بفر_type_host = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cann_host_بفر_type_name,
                           /* .alloc_buffer         = */ ggml_backend_cann_host_بفر_type_alloc_بفر,
                           /* .alloc_buffer_n       = */ عدم,
                           /* .get_alignment        = */ ggml_backend_cpu_بفر_type()->iface.get_alignment,
                           /* .get_max_size         = */ عدم,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_بفر_type()->iface.get_alloc_ناپ,
                           /* .get_alloc_size_n     = */ عدم,
                           /* .is_host              = */ ggml_backend_cpu_بفر_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
        /* .context  = */ خالی_سوچک,
    };

    أرجع &ggml_backend_cann_بفر_type_host;
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
ساكن منطقي ggml_cann_compute_forward(ggml_backend_cann_context & ctx, بنية ggml_tensor * dst) {
    بدل (dst->op) {
        حالة GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            اكسر;
        حالة GGML_OP_GET_صفاں:
            ggml_cann_get_صفاں(ctx, dst);
            اكسر;
        حالة GGML_OP_سیٹ_صفاں:
            ggml_cann_سیٹ_صفاں(ctx, dst);
            اكسر;
        حالة GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            اكسر;
        حالة GGML_OP_جوڑو:
        حالة GGML_OP_ADD1:
            ggml_cann_ثنائی_op<aclnn_جوڑو>(ctx, dst);
            اكسر;
        حالة GGML_OP_SUB:
            ggml_cann_ثنائی_op<aclnn_sub>(ctx, dst);
            اكسر;
        حالة GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            اكسر;
        حالة GGML_OP_MUL:
            ggml_cann_ثنائی_op<aclnn_mul>(ctx, dst);
            اكسر;
        حالة GGML_OP_DIV:
            ggml_cann_ثنائی_op<aclnn_div>(ctx, dst);
            اكسر;
        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(dst)) {
                حالة GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_OP_UNARY(Abs);
                    اكسر;
                حالة GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_OP_UNARY(Neg);
                    اكسر;
                حالة GGML_UNARY_OP_GELU:
                حالة GGML_UNARY_OP_GELU_ERF:
                    // aclnnGelu internally uses the erf-based approximation.
                    GGML_CANN_CALL_OP_UNARY(Gelu);
                    اكسر;
                حالة GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_OP_UNARY(Silu);
                    اكسر;
                حالة GGML_UNARY_OP_GELU_تیز:
                    {
                        خودکار lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                        };
                        ggml_cann_op_unary(lambda, ctx, dst);
                    }
                    اكسر;
                حالة GGML_UNARY_OP_TANH:
                    GGML_CANN_CALL_OP_UNARY(Tanh);
                    اكسر;
                حالة GGML_UNARY_OP_RELU:
                    GGML_CANN_CALL_OP_UNARY(Relu);
                    اكسر;
                حالة GGML_UNARY_OP_SIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Sigmoid);
                    اكسر;
                حالة GGML_UNARY_OP_HARDSIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Hardsigmoid);
                    اكسر;
                حالة GGML_UNARY_OP_HARDSWISH:
                    GGML_CANN_CALL_OP_UNARY(Hardswish);
                    اكسر;
                حالة GGML_UNARY_OP_EXP:
                    GGML_CANN_CALL_OP_UNARY(Exp);
                    اكسر;
                حالة GGML_UNARY_OP_ELU:
                    ggml_cann_elu(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_SGN:
                    GGML_CANN_CALL_OP_UNARY(Sign);
                    اكسر;
                حالة GGML_UNARY_OP_قدم:
                    ggml_cann_قدم(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_SOFTPLUS:
                    ggml_cann_softplus(ctx, dst);
                    اكسر;
                افتراضي:
                    أرجع خطأ;
            }
            اكسر;
        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(dst)) {
                حالة GGML_GLU_OP_REGLU:
                    GGML_CANN_CALL_OP_UNARY_GATED(Relu);
                    اكسر;
                حالة GGML_GLU_OP_GEGLU:
                    ggml_cann_geglu(ctx, dst, 0);  // approximate=0 → tanh
                    اكسر;
                حالة GGML_GLU_OP_GEGLU_ERF:
                    ggml_cann_geglu(ctx, dst, 1);  // approximate=1 → erf
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU:
                    ggml_cann_swiglu(ctx, dst);
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cann_swiglu_clamp(ctx, dst);
                    اكسر;
                حالة GGML_GLU_OP_GEGLU_تیز:
                    ggml_cann_geglu_تیز(ctx, dst);
                    اكسر;
                افتراضي:
                    أرجع خطأ;
            }
            اكسر;
        حالة GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            اكسر;
        حالة GGML_OP_گروہ_NORM:
            ggml_cann_گروہ_norm(ctx, dst);
            اكسر;
        حالة GGML_OP_L2_NORM:
            ggml_cann_l2_norm(ctx, dst);
            اكسر;
        حالة GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cann_cross_entropy_loss(ctx, dst);
            اكسر;
        حالة GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            اكسر;
        حالة GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            اكسر;
        حالة GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            اكسر;
        حالة GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            اكسر;
        حالة GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            اكسر;
        حالة GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            اكسر;
        حالة GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            اكسر;
        حالة GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            اكسر;
        حالة GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            اكسر;
        حالة GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            اكسر;
        حالة GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == خالی_سوچک);
            dst->src[1] = dst->src[0];
            ggml_cann_ثنائی_op<aclnn_mul>(ctx, dst);
            اكسر;
        حالة GGML_OP_SQRT:
            GGML_CANN_CALL_OP_UNARY(Sqrt);
            اكسر;
        حالة GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            اكسر;
        حالة GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            اكسر;
        حالة GGML_OP_سیٹ:
            ggml_cann_سیٹ(ctx, dst);
            اكسر;
        حالة GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            اكسر;
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلو:
        حالة GGML_OP_TRANSPOSE:
            اكسر;
        حالة GGML_OP_DIAG_نقاب_INF:
            ggml_cann_diag_نقاب(ctx, dst, -INFINITY);
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            اكسر;
        حالة GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            اكسر;
        حالة GGML_OP_IM2_خانہ:
            ggml_cann_im2col(ctx, dst);
            اكسر;
        حالة GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            اكسر;
        حالة GGML_OP_جوڑ:
            ggml_cann_جوڑ(ctx, dst);
            اكسر;
        حالة GGML_OP_جوڑ_صفاں:
            ggml_cann_جوڑ_صفاں(ctx, dst);
            اكسر;
        حالة GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            اكسر;
        حالة GGML_OP_ARGMAX:
            ggml_cann_argmax(ctx, dst);
            اكسر;
        حالة GGML_OP_COS:
            ggml_cann_op_unary<aclnn_cos>(ctx, dst);
            اكسر;
        حالة GGML_OP_SIN:
            ggml_cann_op_unary<aclnn_sin>(ctx, dst);
            اكسر;
        حالة GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cann_conv_transpose_1d(ctx, dst);
            اكسر;
        حالة GGML_OP_LOG:
            GGML_CANN_CALL_OP_UNARY(Log);
            اكسر;
        حالة GGML_OP_اوسط:
            ggml_cann_اوسط(ctx, dst);
            اكسر;
        حالة GGML_OP_PAD_REFLECT_1D:
            ggml_cann_pad_reflect_1d(ctx, dst);
            اكسر;
        حالة GGML_OP_گنتی_EQUAL:
            ggml_cann_گنتی_equal(ctx, dst);
            اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            اكسر;
        حالة GGML_OP_باہرلا_PROD:
            ggml_cann_باہرلا_prod(ctx, dst);
            اكسر;
        حالة GGML_OP_GATED_خطی_ATTN:
            ggml_cann_gated_خطی_attn(ctx, dst);
            اكسر;
        حالة GGML_OP_SSM_CONV:
            ggml_cann_ssm_conv(ctx, dst);
            اكسر;
        حالة GGML_OP_CUMSUM:
            ggml_cann_cumsum(ctx, dst);
            اكسر;
        حالة GGML_OP_TRI:
            ggml_cann_tri(ctx, dst);
            اكسر;
        حالة GGML_OP_FILL:
            ggml_cann_fill(ctx, dst);
            اكسر;
        حالة GGML_OP_DIAG:
            ggml_cann_diag(ctx, dst);
            اكسر;
        حالة GGML_OP_SOLVE_TRI:
            ggml_cann_solve_tri(ctx, dst);
            اكسر;
        افتراضي:
            أرجع خطأ;
    }

    أرجع سچ;
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
ساكن ثابت حرف * ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;

    أرجع cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
ساكن فراغ ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    مٹاؤ cann_ctx;
    مٹاؤ backend;
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
ساكن فراغ ggml_backend_cann_سیٹ_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *  tensor,
                                               ثابت فراغ *   انکڑے,
                                               size_t         وتھ,
                                               size_t         size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_بفر_t       buf      = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    GGML_ASSERT(buf->buft == ggml_backend_cann_بفر_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync((حرف *) tensor->انکڑے + وتھ, size, انکڑے, size, ACL_MEMCPY_HOST_منزل_DEVICE,
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
ساكن فراغ ggml_backend_cann_get_tensor_async(ggml_backend_t      backend,
                                               ثابت ggml_tensor * tensor,
                                               فراغ *              انکڑے,
                                               size_t              وتھ,
                                               size_t              size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_بفر_t       buf      = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    GGML_ASSERT(buf->buft == ggml_backend_cann_بفر_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync(انکڑے, size, (حرف *) tensor->انکڑے + وتھ, size, ACL_MEMCPY_DEVICE_منزل_HOST,
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
ساكن منطقي ggml_backend_cann_cpy_tensor_async(ggml_backend_t      backend_src,
                                               ggml_backend_t      backend_dst,
                                               ثابت ggml_tensor * src,
                                               ggml_tensor *       dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) || ggml_backend_is_cann(backend_dst));

    GGML_ASSERT(!is_matmul_بھار((ثابت ggml_tensor *) src));

    إذا (!ggml_backend_buft_is_cann(src->بفر->buft) || !ggml_backend_buft_is_cann(dst->بفر->buft)) {
        أرجع خطأ;
    }

    ggml_backend_بفر_t buf_src = src->view_src ? src->view_src->بفر : src->بفر;
    ggml_backend_بفر_t buf_dst = dst->view_src ? dst->view_src->بفر : dst->بفر;

    ggml_backend_cann_context * cann_ctx_src = (ggml_backend_cann_context *) backend_src->context;
    ggml_backend_cann_context * cann_ctx_dst = (ggml_backend_cann_context *) backend_dst->context;

    size_t copy_ناپ = ggml_nbytes(dst);
    إذا (copy_ناپ == 0) {
        أرجع سچ;
    }
    إذا (backend_src != backend_dst) {
#جے_تعریف_شدہ ASCEND_310P
        // TODO: Support 310p P2P copy
        أرجع خطأ;
#جے_انت
        ggml_backend_cann_بفر_context * buf_ctx_src = (ggml_backend_cann_بفر_context *) buf_src->context;
        ggml_backend_cann_بفر_context * buf_ctx_dst = (ggml_backend_cann_بفر_context *) buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device, cann_ctx_dst->device));
        إذا (!canAccessPeer) {
            أرجع خطأ;
        }

        // need open both directions for memcpyasync between devices.
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_سیٹ_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        // wait for task_queue empty to keep task order.
        ACL_CHECK(aclrtMemcpyAsync(dst->انکڑے, copy_ناپ, src->انکڑے, copy_ناپ, ACL_MEMCPY_DEVICE_منزل_DEVICE,
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
        ACL_CHECK(aclrt_Synchronize_دھارا(cann_ctx_src->stream()));
    } وإلا {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->انکڑے, copy_ناپ, src->انکڑے, copy_ناپ, ACL_MEMCPY_DEVICE_منزل_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    أرجع سچ;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
ساكن فراغ ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_سیٹ_device(cann_ctx->device);
    ACL_CHECK(aclrt_Synchronize_دھارا(cann_ctx->stream()));
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
ساكن منطقي ggml_cann_can_fuse(ثابت بنية ggml_cgraph *          cgraph,
                               صحيح                                 گرہ_idx,
                               std::initializer_list<تعداد ggml_op> ops) {
    إذا (!ggml_can_fuse(cgraph, گرہ_idx, ops)) {
        أرجع خطأ;
    }

    // CANN backend supports fusing ADD + RMS_NORM operations
    إذا ((ops.size() == 2) && ops.begin()[0] == GGML_OP_جوڑو && ops.begin()[1] == GGML_OP_RMS_NORM) {
        ggml_tensor * جوڑو_گرہ = cgraph->nodes[گرہ_idx];
        // TODO: support broadcast for ADD + RMS_NORM
        إذا (جوڑو_گرہ->src[0]->ne[0] != جوڑو_گرہ->src[1]->ne[0] || جوڑو_گرہ->src[0]->ne[1] != جوڑو_گرہ->src[1]->ne[1] ||
            جوڑو_گرہ->src[0]->ne[2] != جوڑو_گرہ->src[1]->ne[2] || جوڑو_گرہ->src[0]->ne[3] != جوڑو_گرہ->src[1]->ne[3]) {
            أرجع خطأ;
        }
        أرجع سچ;
    }

    أرجع خطأ;
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
ساكن فراغ evaluate_and_capture_cann_گراف(ggml_backend_cann_context * cann_ctx,
                                            ggml_cgraph *               cgraph,
                                            منطقي                        use_cann_گراف,
                                            منطقي                        cann_گراف_capture_required) {
#جے_تعریف_شدہ USE_ACL_گراف
    إذا (use_cann_گراف && cann_گراف_capture_required) {  // Begin CANN graph capture
        ACL_CHECK(aclmdlRICaptureBegin(cann_ctx->stream(), ACL_MODEL_RI_CAPTURE_منوال_GLOBAL));
    }
#جے_انت  // USE_ACL_GRAPH
    // Only perform the graph execution if CANN graphs are not enabled, or we are capturing the graph.
    // With the use of CANN graphs, the execution will be performed by the graph launch.
    ساكن منطقي opt_fusion = parse_bool(get_env_as_lowercase("GGML_CANN_OPERATOR_FUSION").value_or(""));

    إذا (!use_cann_گراف || cann_گراف_capture_required) {
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * گرہ = cgraph->nodes[i];
            إذا (opt_fusion) {
                إذا (ggml_cann_can_fuse(cgraph, i, { GGML_OP_جوڑو, GGML_OP_RMS_NORM })) {
                    ggml_cann_op_جوڑو_rms_norm_fused(*cann_ctx, گرہ, cgraph->nodes[i + 1]);
                    i++;
                    استمر;
                }
            }

            إذا (ggml_is_خالی_تھاواں(گرہ) || گرہ->op == GGML_OP_RESHAPE || گرہ->op == GGML_OP_TRANSPOSE ||
                گرہ->op == GGML_OP_VIEW || گرہ->op == GGML_OP_ترتیب_بدلو || گرہ->op == GGML_OP_NONE) {
                استمر;
            }

            إذا ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                استمر;
            }

            منطقي ok = ggml_cann_compute_forward(*cann_ctx, گرہ);
            إذا (!ok) {
                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, گرہ->name, ggml_op_name(گرہ->op));
            }
            GGML_ASSERT(ok);
        }
    }

#جے_تعریف_شدہ USE_ACL_گراف
    إذا (use_cann_گراف) {
        GGML_ASSERT(!cann_ctx->گراف_lru_cache.cache_list.خالی_تھاواں());
        ggml_cann_گراف * matched_گراف = cann_ctx->گراف_lru_cache.cache_list.front();

        إذا (cann_گراف_capture_required) {  // End CANN graph capture
            ACL_CHECK(aclmdl_RICapture_سرا(cann_ctx->stream(), &matched_گراف->graph));
        }

        // Execute CANN graph
        ACL_CHECK(aclmdlRIExecuteAsync(matched_گراف->graph, cann_ctx->stream()));
    }
#جے_انت  // USE_ACL_GRAPH
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
ساكن تعداد ggml_حالت ggml_backend_cann_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_سیٹ_device(cann_ctx->device);
    g_nz_workspaces[cann_ctx->device].clear();

    // calculate rope cache for fist layer in current device.
    cann_ctx->rope_cache.cached = خطأ;

    منطقي گراف_capture_required = خطأ;
#جے_تعریف_شدہ USE_ACL_گراف
    منطقي use_cann_گراف = سچ;

    ساكن منطقي prefill_use_گراف = parse_bool(get_env_as_lowercase("GGML_CANN_PREFILL_USE_GRAPH").value_or(""));
    إذا (!prefill_use_گراف) {
        // Do not use acl_graph for prefill.
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * گرہ = cgraph->nodes[i];
            // TODO: Optimize here. Currently, we can only
            // get seq_len by FA's input.
            إذا (گرہ->op == GGML_OP_FLASH_ATTN_EXT) {
                // Q -> src[0], shape: [B, S, N, D]
                use_cann_گراف = (گرہ->src[0]->ne[1] == 1);
                اكسر;
            }
        }
    }

    إذا (!cann_ctx->acl_گراف_منوال) {
        use_cann_گراف = خطأ;
    }

    إذا (use_cann_گراف) {
        // If no matching graph is found, the graph needs to be recaptured.
        گراف_capture_required = !cann_ctx->گراف_lru_cache.لبھو_and_move_منزل_front(cgraph);

        إذا (گراف_capture_required) {
            // If no matching graph is found, add a new ACL graph.
            ggml_cann_گراف * new_گراف = ggml_cann_گراف::create_from_cgraph(cgraph);
            cann_ctx->گراف_lru_cache.ڈالو(new_گراف);

            // Pre-load rope cache before graph capture.  During capture the
            // stream cannot perform host-to-device memcpy or device memory
            // malloc/free.  Running the full cache init now populates the
            // cache metadata so these branches are skipped during capture,
            // while also warming up the memory pool.
            لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * گرہ = cgraph->nodes[i];
                إذا (گرہ->op == GGML_OP_ROPE) {
                    ggml_cann_rope_cache_preload(*cann_ctx, گرہ);
                    اكسر;
                }
            }
        }
    }
#وإلا
    منطقي use_cann_گراف = خطأ;
#جے_انت  // USE_ACL_GRAPH
    evaluate_and_capture_cann_گراف(cann_ctx, cgraph, use_cann_گراف, گراف_capture_required);

    أرجع GGML_حالت_SUCCESS;
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
ساكن منطقي ggml_backend_cann_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    بدل (op->op) {
        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(op)) {
                حالة GGML_UNARY_OP_ABS:
                حالة GGML_UNARY_OP_NEG:
                حالة GGML_UNARY_OP_GELU:
                حالة GGML_UNARY_OP_SILU:
                حالة GGML_UNARY_OP_RELU:
                حالة GGML_UNARY_OP_SIGMOID:
                حالة GGML_UNARY_OP_HARDSIGMOID:
                حالة GGML_UNARY_OP_HARDSWISH:
                حالة GGML_UNARY_OP_GELU_تیز:
                حالة GGML_UNARY_OP_TANH:
                حالة GGML_UNARY_OP_EXP:
                حالة GGML_UNARY_OP_ELU:
                حالة GGML_UNARY_OP_SGN:
                حالة GGML_UNARY_OP_قدم:
                حالة GGML_UNARY_OP_GELU_ERF:
                حالة GGML_UNARY_OP_SOFTPLUS:
                    أرجع سچ;
                افتراضي:
                    أرجع خطأ;
            }
        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(op)) {
                حالة GGML_GLU_OP_REGLU:
                حالة GGML_GLU_OP_GEGLU:
                حالة GGML_GLU_OP_SWIGLU:
                حالة GGML_GLU_OP_GEGLU_ERF:
                حالة GGML_GLU_OP_GEGLU_تیز:
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    أرجع سچ;
                افتراضي:
                    أرجع خطأ;
            }
            اكسر;
        حالة GGML_OP_MUL_MAT:
            {
                بدل (op->src[0]->type) {
#جے_بنا_تعریف ASCEND_310P
                    حالة GGML_TYPE_BF16:
#جے_انت
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_F32:
                        أرجع سچ;
                    حالة GGML_TYPE_Q8_0:
                    حالة GGML_TYPE_Q4_0:
#جے_تعریف_شدہ ASCEND_310P
                        // Q4 && Q8 per group is not support on 310p device
                        أرجع خطأ;
#وإلا
                        // only support contiguous for quantized types.
                        أرجع ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#جے_انت
                    افتراضي:
                        أرجع خطأ;
                }
            }
        حالة GGML_OP_MUL_MAT_ID:
            بدل (op->src[0]->type) {
                حالة GGML_TYPE_F16:
                حالة GGML_TYPE_F32:
                    أرجع سچ;
                حالة GGML_TYPE_Q8_0:
                حالة GGML_TYPE_Q4_0:
#جے_تعریف_شدہ ASCEND_310P
                    // Q4 && Q8 per group is not support on 310p device
                    أرجع خطأ;
#وإلا
                    // only support contiguous for quantized types.
                    أرجع ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#جے_انت
                افتراضي:
                    أرجع خطأ;
            }
        // embedding
        حالة GGML_OP_GET_صفاں:
            {
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
#جے_بنا_تعریف ASCEND_310P
                    حالة GGML_TYPE_BF16:
#جے_انت
                    حالة GGML_TYPE_Q8_0:
                        أرجع سچ;
                    افتراضي:
                        أرجع خطأ;
                }
            }
            اكسر;
        حالة GGML_OP_سیٹ_صفاں:
            {
                بدل (op->type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
#جے_بنا_تعریف ASCEND_310P
                    حالة GGML_TYPE_BF16:
#جے_انت
                        أرجع سچ;
                    افتراضي:
                        أرجع خطأ;
                }
            }
            اكسر;
        حالة GGML_OP_CPY:
            {
                ggml_tensor * src = op->src[0];
#جے_تعریف_شدہ ASCEND_310P
                إذا ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                    // only support F32 and F16 on 310P.
                    أرجع خطأ;
                }
#وإلا
                إذا ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
                    // only support F32, F16 and BF16.
                    أرجع خطأ;
                }
#جے_انت
                أرجع سچ;
            }
            اكسر;
        حالة GGML_OP_CONT:
            {
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_F16:
#جے_بنا_تعریف ASCEND_310P
                    حالة GGML_TYPE_BF16:
#جے_انت
                        أرجع سچ;
                    افتراضي:
                        أرجع خطأ;
                }
            }
        حالة GGML_OP_ROPE:
            {
                إذا (((ثابت int32_t *) op->op_params)[15] != 0) {
                    أرجع خطأ; // FIXME: support ggml_rope_set_offset
                }
                إذا (op->src[0]->ne[0] > 896) {
                    أرجع خطأ;
                }
#جے_تعریف_شدہ ASCEND_310P
                // TODO: Support rope_dim < ne00(dim)
                إذا (op->src[0]->ne[0] != op->op_params[1]) {
                    أرجع خطأ;
                }
                إذا (!ggml_is_contiguous(op->src[0])) {
                    أرجع خطأ;
                }
#جے_انت
                أرجع سچ;
            }
        حالة GGML_OP_UPSCALE:
            {
                // aclnnUpsampleNearest2dGetWorkspaceSize not support
                // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
                إذا (op->src[0]->ne[2] * op->ne[3] != op->src[0]->ne[3] * op->ne[2]) {
                    أرجع خطأ;
                }
                إذا (op->op_params[0] != GGML_SCALE_منوال_سبھ_توں_نیڑے) {
                    أرجع خطأ;
                }
                إذا (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                    أرجع خطأ;
                }
                أرجع سچ;
            }
        حالة GGML_OP_POOL_2D:
            {
                ثابت int32_t * opts = (ثابت int32_t *) op->op_params;
#جے_تعریف_شدہ ASCEND_310P
                تعداد ggml_op_pool opt = static_cast<ggml_op_pool>(opts[0]);
                إذا (opt == GGML_OP_POOL_MAX) {
                    أرجع خطأ;
                }
#جے_انت
                ثابت صحيح k0 = opts[1];
                ثابت صحيح k1 = opts[2];
                ثابت صحيح p0 = opts[5];
                ثابت صحيح p1 = opts[6];
                // value of paddingH should be at most half of kernelH
                // value of paddingW should be at most half of kernelW
                أرجع (p0 <= (k0 / 2)) && (p1 <= (k1 / 2));
            }
        حالة GGML_OP_جوڑ:
            أرجع ggml_is_contiguous_صفاں(op->src[0]);
        حالة GGML_OP_L2_NORM:
        حالة GGML_OP_CROSS_ENTROPY_LOSS:
        حالة GGML_OP_DUP:
        حالة GGML_OP_IM2_خانہ:
        حالة GGML_OP_CONCAT:
        حالة GGML_OP_REPEAT:
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلو:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_NORM:
        حالة GGML_OP_جوڑو:
        حالة GGML_OP_ADD1:
        حالة GGML_OP_SUB:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
        حالة GGML_OP_RMS_NORM:
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_DIAG_نقاب_INF:
        حالة GGML_OP_جوڑ_صفاں:
        حالة GGML_OP_ARGSORT:
        حالة GGML_OP_ACC:
        حالة GGML_OP_سیٹ:
        حالة GGML_OP_گروہ_NORM:
            أرجع سچ;
        حالة GGML_OP_PAD:
            // TODO: add circular padding support for cann, see https://github.com/ggml-org/llama.cpp/pull/16985
            أرجع ggml_get_op_params_i32(op, 8) == 0;
        حالة GGML_OP_ARANGE:
        حالة GGML_OP_TIMESTEP_EMBEDDING:
        حالة GGML_OP_LEAKY_RELU:
        حالة GGML_OP_ARGMAX:
        حالة GGML_OP_COS:
        حالة GGML_OP_SIN:
        حالة GGML_OP_LOG:
        حالة GGML_OP_اوسط:
        حالة GGML_OP_PAD_REFLECT_1D:
        حالة GGML_OP_گنتی_EQUAL:
        حالة GGML_OP_GATED_خطی_ATTN:
            أرجع سچ;
        حالة GGML_OP_باہرلا_PROD:
            {
#جے_تعریف_شدہ ASCEND_310P
                // Ger is not supported on 310p device
                أرجع خطأ;
#وإلا
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_F32:
                        أرجع سچ;
                    افتراضي:
                        أرجع خطأ;
                }
#جے_انت
            }
        حالة GGML_OP_CONV_TRANSPOSE_1D:
            أرجع سچ;
        حالة GGML_OP_SCALE:
            عائم bias;
            حافظہ_نقل(&bias, (ثابت عائم *) (op->op_params) + 1, حجم(عائم));
            أرجع bias == 0.0f;  // TODO: support bias != 0.0f
        حالة GGML_OP_SOFT_MAX:
            // TODO: support attention sinks [TAG_ATTN_SINKS]
            إذا (op->src[2]) {
                أرجع خطأ;
            }
            أرجع سچ;
        حالة GGML_OP_FLASH_ATTN_EXT:
            {
#جے_تعریف_شدہ ASCEND_310P
                // FA not support on 310p device
                أرجع خطأ;
#وإلا
                // derived from [ggml-cuda.cu]
                إذا (op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16) {
                    أرجع خطأ;
                }
                إذا (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32 &&
                    op->src[1]->type != GGML_TYPE_BF16) {
                    أرجع خطأ;
                }
                إذا (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16) {
                    أرجع خطأ;
                }
                // TODO: support attention sinks [TAG_ATTN_SINKS]
                إذا (op->src[4]) {
                    أرجع خطأ;
                }
                إذا (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                    // different head sizes of K and V are not supported yet
                    أرجع خطأ;
                }
                عائم logitSoftcap = 0.0f;
                حافظہ_نقل(&logitSoftcap, (ثابت عائم *) (op->op_params) + 2, حجم(عائم));
                إذا (logitSoftcap != 0.0f) {
                    أرجع خطأ;
                }
                أرجع سچ;
#جے_انت
            }
        حالة GGML_OP_SSM_CONV:
            أرجع سچ;
        حالة GGML_OP_CUMSUM:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_TRI:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_FILL:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_DIAG:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_SOLVE_TRI:
            أرجع op->src[0]->type == GGML_TYPE_F32;
        افتراضي:
            أرجع خطأ;
    }

    GGML_بن_ورتیا(dev);
}

/**
 * @brief Records an event on the CANN backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
ساكن فراغ ggml_backend_cann_event_ریکارڈ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrt_ریکارڈ_Event((aclrtEvent) event->context, cann_ctx->stream()));
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
ساكن فراغ ggml_backend_cann_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    إذا (ggml_backend_is_cann(backend)) {
        ACL_CHECK(aclrt_دھارا_Wait_Event(cann_ctx->stream(), (aclrtEvent) event->context));
    } وإلا {
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
ساكن ثابت ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_cann_گراف_compute,
    /* .event_record            = */ ggml_backend_cann_event_ریکارڈ,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .graph_optimize          = */ عدم,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
ساكن ggml_guid_t ggml_backend_cann_guid() {
    ساكن ggml_guid guid = { 0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                              0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64 };
    أرجع &guid;
}

// backend device
بنية ggml_backend_cann_device_context {
    صحيح         device;
    std::string name;
    std::string description;
    صحيح op_offload_min_batch_ناپ;
};

ساكن ثابت حرف * ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    أرجع ctx->name.c_str();
}

ساكن ثابت حرف * ggml_backend_cann_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    أرجع ctx->description.c_str();
}

ساكن فراغ ggml_backend_cann_device_get_حافظہ(ggml_backend_dev_t dev, size_t * حرر, size_t * کل) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ggml_backend_cann_get_device_حافظہ(ctx->device, حرر, کل);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_cann_device_get_type(ggml_backend_dev_t dev) {
    GGML_بن_ورتیا(dev);
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساكن فراغ ggml_backend_cann_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type        = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);

    منطقي host_بفر = getenv("GGML_CANN_NO_PINNED") == خالی_سوچک;

    props->caps = {
        /* .async                 = */ خطأ,
        /* .host_buffer           = */ host_بفر,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ سچ,
        /* .mmap_support          = */ سچ,
    };
}

ساكن ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_بن_ورتیا(params);
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    أرجع ggml_backend_cann_init(ctx->device);
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
ساكن منطقي ggml_backend_cann_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    إذا (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context *      dev_ctx  = (ggml_backend_cann_device_context *) dev->context;
        ggml_backend_cann_بفر_type_context * buft_ctx = (ggml_backend_cann_بفر_type_context *) buft->context;
        أرجع buft_ctx->device == dev_ctx->device;
    }
    أرجع خطأ;
}

ساكن ggml_backend_بفر_type_t ggml_backend_cann_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    أرجع ggml_backend_cann_بفر_type(ctx->device);
}

ساكن ggml_backend_بفر_type_t ggml_backend_cann_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    GGML_بن_ورتیا(dev);
    أرجع ggml_backend_cann_host_بفر_type();
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
ساكن منطقي ggml_backend_cann_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;

    أرجع op->ne[1] >= dev_ctx->op_offload_min_batch_ناپ && op->op != GGML_OP_GET_صفاں;
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
ساكن ggml_backend_event_t ggml_backend_cann_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *) dev->context;

    ggml_cann_سیٹ_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    أرجع جديد ggml_backend_event{
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
ساكن فراغ ggml_backend_cann_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent) event->context));

    مٹاؤ event;
    GGML_بن_ورتیا(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
ساكن فراغ ggml_backend_cann_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent) event->context));

    GGML_بن_ورتیا(dev);
}

ساكن ثابت ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_حافظہ,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,  // called for every card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_بفر_type,
    /* .get_host_buffer_type    = */ ggml_backend_cann_device_get_host_بفر_type,
    /* .buffer_from_host_ptr    = */ عدم,  // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};

// backend reg
بنية ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

ساكن ثابت حرف * ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_بن_ورتیا(reg);
    أرجع GGML_CANN_NAME;
}

ساكن size_t ggml_backend_cann_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    أرجع ctx->devices.size();
}

ساكن ggml_backend_dev_t ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    GGML_ASSERT(اشاریہ < ctx->devices.size());
    أرجع ctx->devices[اشاریہ];
}

ساكن فراغ * ggml_backend_cann_reg_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_بن_ورتیا(reg);
    GGML_بن_ورتیا(name);
    // reserved for future use
    أرجع خالی_سوچک;
}

ساكن ثابت ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_گنتی,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_پتہ,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    ساكن ggml_backend_reg reg;
    ساكن منطقي             initialized = خطأ;

    {
        ساكن std::mutex           mutex;
        std::lock_guard<std::mutex> جندرا(mutex);
        إذا (!initialized) {
            aclInit(خالی_سوچک);
            ggml_backend_cann_reg_context * ctx = جديد ggml_backend_cann_reg_context;
            ثابت صحيح min_batch_ناپ = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            لكل (صحيح i = 0; i < ggml_cann_info().device_گنتی; i++) {
                ggml_backend_cann_device_context * dev_ctx = جديد ggml_backend_cann_device_context();
                dev_ctx->description                       = aclrtGetSocName();
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_CANN_NAME + std::to_string(i);
                dev_ctx->op_offload_min_batch_ناپ         = min_batch_ناپ;
                ggml_cann_سیٹ_device(i);
                ggml_backend_dev_t dev = جديد ggml_backend_device{ /* .iface   = */ ggml_backend_cann_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_نسخہ,
                                    /* .iface       = */ ggml_backend_cann_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = سچ;
    }

    أرجع &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(خالی_سوچک);
    إذا (device < 0 || device >= ggml_backend_cann_get_device_گنتی()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        أرجع خالی_سوچک;
    }

    ggml_backend_cann_context * ctx = جديد ggml_backend_cann_context(device);
    إذا (ctx == خالی_سوچک) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        أرجع خالی_سوچک;
    }
    ggml_cann_سیٹ_device(ctx->device);
    ggml_backend_t cann_backend =
        جديد ggml_backend{ /* .guid      = */ ggml_backend_cann_guid(),
                          /* .interface = */ ggml_backend_cann_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                          /* .context   = */ ctx };

    أرجع cann_backend;
}

منطقي ggml_backend_is_cann(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_گنتی() {
    أرجع ggml_cann_info().device_گنتی;
}

فراغ ggml_backend_cann_get_device_description(int32_t device, حرف * description, size_t description_ناپ) {
    ggml_cann_سیٹ_device(device);
    ثابت حرف * soc_name = aclrtGetSocName();
    snprintf(description, description_ناپ, "%s", soc_name);
}

فراغ ggml_backend_cann_get_device_حافظہ(int32_t device, size_t * حرر, size_t * کل) {
    ggml_cann_سیٹ_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, حرر, کل));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)

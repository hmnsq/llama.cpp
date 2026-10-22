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

#समावेश "ggml-cann.h"

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-cann/aclnn_ops.h"
#समावेश "ggml-cann/common.h"
#समावेश "ggml-impl.h"
#समावेश "ggml.h"

#समावेश <acl/acl.h>
#समावेश <aclnnop/aclnn_trans_matmul_weight.h>
#समावेश <stdarg.h>

#समावेश <chrono>
#समावेश <cmath>
#समावेश <cstdio>
#समावेश <cstring>
#समावेश <memory>
#समावेश <mutex>
#समावेश <optional>
#समावेश <queue>
#समावेश <unordered_map>
#समावेश <unordered_set>
#समावेश <vector>

#घोषणा GGML_COMMON_DECL_C

#समावेश "ggml-common.h"

#घोषणा GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] व्योम ggml_cann_error(स्थिर अक्षर * stmt, स्थिर अक्षर * func, स्थिर अक्षर * file, पूर्णांक पंक्ति, स्थिर अक्षर * संदेश) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", संदेश);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, पंक्ति);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

// Thread-local variable to record the current device of this thread.
thread_local पूर्णांक g_current_cann_device = -1;

/**
 * @brief Set the CANN device to be used.
 *
 * @param device The target device ID to set.
 */
व्योम ggml_cann_set_device(स्थिर int32_t device) {
    // int current_device = -1;
    // Note: In some CANN versions, if no device has been set yet,
    //       aclrtGetDevice(&current_device) may return 0 by default.
    // aclrtGetDevice(&current_device);

    // If the current device is already the target one, no need to switch.
    अगर (device == g_current_cann_device) {
        वापस;
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
std::optional<std::string> get_env_as_lowercase(स्थिर std::string & name) {
    स्थिर अक्षर * val = std::दो_पर्या(name.c_str());
    अगर (!val) {
        वापस std::nullopt;
    }
    std::string res = std::string(val);
    std::transform(res.begin(), res.end(), res.begin(), ::छोटे);
    वापस res;
}

/**
 * @brief Verify whether the environment variable is a valid value.
 */
bool parse_bool(स्थिर std::string & value) {
    अटल स्थिर std::unordered_set<std::string> valid_मूल्यावली = { "on", "1", "yes", "y", "enable", "true" };
    वापस valid_मूल्यावली.find(value) != valid_मूल्यावली.end();
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
पूर्णांक parse_integer(स्थिर std::string & value) {
    प्रयास {
        वापस std::stoi(value);
    } पकड़ो (...) {
        वापस 0;
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
अटल ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrt_Get_Device_गिनती((uint32_t *) &info.device_गिनती);

    अगर (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n", __func__, acl_Get_Recent_Err_संदेश());
        वापस info;
    }

    GGML_ASSERT(info.device_गिनती <= GGML_CANN_MAX_DEVICES);

    क्रम (पूर्णांक id = 0; id < info.device_गिनती; ++id) {
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

        माप_प्रकार मुक्त, योग;
        ggml_backend_cann_get_device_memory(id, &मुक्त, &योग);
        info.devices[id].योग_vram = मुक्त;
    }

    // TODO: add more device info later.
    वापस info;
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
स्थिर ggml_cann_device_info & ggml_cann_info() {
    अटल ggml_cann_device_info info = ggml_cann_init();
    वापस info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(priority segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
काष्ठा ggml_cann_pool_buf_prio : खुला ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    अटल स्थिर माप_प्रकार max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    अटल स्थिर माप_प्रकार min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    अटल स्थिर माप_प्रकार alignment = 128;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    पूर्णांक device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    bool disable_clean = false;

    /**
     * @brief Structure representing a CANN buffer.
     */
    काष्ठा ggml_cann_बफर {
        व्योम *                                ptr  = nullptr;  ///< Pointer to the buffer.
        माप_प्रकार                                size = 0;        ///< Size of the buffer.
        std::chrono::steady_clock::time_point last_used;       ///< Last used time.

        bool चालक>(स्थिर ggml_cann_बफर & other) स्थिर { वापस size > other.size; }
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    std::unordered_map<व्योम *, माप_प्रकार>                                                   बफर_pool;
    std::priority_queue<ggml_cann_बफर, std::vector<ggml_cann_बफर>, std::greater<>> free_buffers;

    /**
     * @brief Total size of all buffers in the pool.
     */
    माप_प्रकार pool_आकार = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_buf_prio(पूर्णांक device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf_prio() {
        ggml_cann_set_device(device);
        क्रम (स्वतः & [b_ptr, b_आकार] : बफर_pool) {
            aclrtFree(b_ptr);
            pool_आकार -= b_आकार;
        }
        बफर_pool.सफा();
        GGML_ASSERT(pool_आकार == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    व्योम * alloc(माप_प्रकार size, माप_प्रकार * actual_आकार) override {
        size = GGML_PAD(size, alignment);
        अगर (size == 0) {
            size = alignment;
        }

        व्योम * ptr = nullptr;
        स्वतः   now = std::chrono::steady_clock::now();

        std::vector<ggml_cann_बफर> free_buffers_rest;
        free_buffers_rest.reserve(free_buffers.size());
        जबतक (!free_buffers.empty()) {
            स्वतः b = free_buffers.top();
            free_buffers.निकालो();

            अगर (b.size >= size) {
                // reuse the buffer if the size is enough
                स्थिर माप_प्रकार margin = b.size - size;
                अगर (margin <= max_reuse_margin) {
                    *actual_आकार = b.size;
                    ptr          = b.ptr;
#अगर_घोषित DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#पूर्ण_अगर
                    अवरोध;
                }
            }

            bool should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            अगर (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_आकार -= b.size;
                बफर_pool.erase(b.ptr);
#अगर_घोषित DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#पूर्ण_अगर
                जारी;
            }
            free_buffers_rest.push_back(b);
        }
        क्रम (ggml_cann_बफर & b : free_buffers_rest) {
            free_buffers.धकेलो(std::move(b));
        }

#अगर_घोषित DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d] free pool_size = %5u MB\n\n", device,
                      (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576));
#पूर्ण_अगर
        अगर (ptr != nullptr) {
            वापस ptr;
        }

        // allocate a new buffer if no buffer can be reused
        ggml_cann_set_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        *actual_आकार = size;
        pool_आकार += size;
#अगर_घोषित DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: allocate %p, "
            "pool_size = %5u MB, "
            "size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576),
            (uint32_t) (GGML_PAD(size, 1048576) / 1048576));
#पूर्ण_अगर
        बफर_pool.emplace(ptr, size);
        वापस ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    व्योम मुक्त(व्योम * ptr, माप_प्रकार size) override {
        GGML_UNUSED(size);
        स्वतः it = बफर_pool.find(ptr);
        अगर (it == बफर_pool.end()) {
            GGML_ABORT("cann pool[%d]: buffer %p not found in pool\n", device, ptr);
        }

        स्वतः now = std::chrono::steady_clock::now();
        free_buffers.emplace(ggml_cann_बफर{ ptr, it->second, now });
#अगर_घोषित DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: return   %p, "
            "pool_size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576));
#पूर्ण_अगर
    }
};

/**
 * @brief A pool of CANN buffers(segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
काष्ठा ggml_cann_pool_buf : खुला ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    अटल स्थिर माप_प्रकार max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    अटल स्थिर माप_प्रकार min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    अटल स्थिर माप_प्रकार alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    अटल स्थिर पूर्णांक MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    पूर्णांक device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    bool disable_clean = false;

    /**
     * @brief Structure representing a CANN buffer.
     */
    काष्ठा ggml_cann_बफर {
        व्योम *                                ptr  = nullptr;  ///< Pointer to the buffer memory.
        माप_प्रकार                                size = 0;        ///< Size of the buffer.
        bool                                  used = false;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point last_used;       ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_बफर बफर_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    माप_प्रकार pool_आकार = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_buf(पूर्णांक device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf() {
        ggml_cann_set_device(device);
        क्रम (पूर्णांक i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_बफर & b = बफर_pool[i];
            अगर (b.ptr != nullptr) {
                aclrtFree(b.ptr);
                pool_आकार -= b.size;
            }
        }
        GGML_ASSERT(pool_आकार == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    व्योम * alloc(माप_प्रकार size, माप_प्रकार * actual_आकार) override {
        size = GGML_PAD(size, alignment);
        अगर (size == 0) {
            size = alignment;
        }

        व्योम * ptr = nullptr;
        स्वतः   now = std::chrono::steady_clock::now();

        पूर्णांक i = 0;
        क्रम (; i < MAX_BUFFERS; ++i) {
            ggml_cann_बफर & b = बफर_pool[i];
            अगर (b.ptr == nullptr) {
                अवरोध;
            }
            अगर (b.used) {
                जारी;
            }
            अगर (b.size >= size) {
                // reuse the buffer if the size is enough
                स्थिर माप_प्रकार margin = b.size - size;
                अगर (margin <= max_reuse_margin) {
                    *actual_आकार = b.size;
                    b.used       = true;
                    ptr          = b.ptr;
#अगर_घोषित DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#पूर्ण_अगर
                    अवरोध;
                }
            }

            bool should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            अगर (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_आकार -= b.size;
#अगर_घोषित DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#पूर्ण_अगर
                b.ptr = nullptr;
            }
        }
        अगर (ptr != nullptr) {
            वापस ptr;
        }

        अगर (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_cann_बफर & b = बफर_pool[i];
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtMalloc(&b.ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
            pool_आकार += size;
            *actual_आकार = size;
            b.size       = size;
            b.used       = true;
            अगर (i >= MAX_BUFFERS - 8) {
                GGML_LOG_WARN("cann pool[%d]: slots almost full\n", device);
            }
#अगर_घोषित DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576),
                (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#पूर्ण_अगर
            वापस b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    व्योम मुक्त(व्योम * ptr, माप_प्रकार size) override {
        GGML_UNUSED(size);
        क्रम (पूर्णांक i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_बफर & b = बफर_pool[i];
            अगर (b.ptr != ptr) {
                जारी;
            }
            b.used      = false;
            b.last_used = std::chrono::steady_clock::now();
#अगर_घोषित DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_आकार, 1048576) / 1048576));
#पूर्ण_अगर
            वापस;
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
काष्ठा ggml_cann_pool_vmm : खुला ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    माप_प्रकार max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    पूर्णांक device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    व्योम * pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    माप_प्रकार pool_used = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    माप_प्रकार pool_आकार = 0;

    /**
     * @brief Allocation granularity for the virtual memory pool.
     */
    माप_प्रकार granularity;

    /**
     * @brief Handles for the physical memory allocated.
     */
    std::vector<aclrtDrvMemHandle> handles;

    /**
     * @brief Offsets for the mapped memory regions.
     */
    std::vector<व्योम *> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_vmm(पूर्णांक device) : device(device) {
        स्वतः dev    = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size    = dev.योग_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        अगर (pool_addr != 0) {
            क्रम (स्वतः & offset : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(offset));
            }
            क्रम (स्वतः & handle : handles) {
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
    व्योम * alloc(माप_प्रकार size, माप_प्रकार * actual_आकार) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        स्थिर माप_प्रकार alignment = 128;
        size                   = GGML_PAD(size, alignment);
        अगर (size == 0) {
            size = alignment;
        }

        माप_प्रकार avail = pool_आकार - pool_used;

        अगर (size > avail) {
            // round up to the next multiple of the granularity
            माप_प्रकार reserve_आकार = size - avail;
            reserve_आकार        = GGML_PAD(reserve_आकार, granularity);

            GGML_ASSERT(pool_आकार + reserve_आकार <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType       = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr              = ACL_HBM_MEM_HUGE;
            prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id          = device;
            prop.reserve              = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_आकार, &prop, 0));

            // reserve virtual address space (if not already reserved)
            अगर (pool_addr == 0) {
                ACL_CHECK(aclrtReserveMemAddress(&pool_addr, max_size, 0, शून्य, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((अक्षर *) pool_addr + pool_आकार, reserve_आकार, 0, handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((अक्षर *) pool_addr + pool_आकार);

            // add to the pool
            pool_आकार += reserve_आकार;

#अगर_घोषित DEBUG_CANN_MALLOC
            GGML_LOG_INFO("cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n", device,
                          (अचिन्हित दीर्घ दीर्घ) (pool_आकार / 1024 / 1024),
                          (अचिन्हित दीर्घ दीर्घ) (reserve_आकार / 1024 / 1024));
#पूर्ण_अगर
        }

        GGML_ASSERT(pool_addr != 0);

        व्योम * ptr   = (व्योम *) ((अक्षर *) pool_addr + pool_used);
        *actual_आकार = size;
        pool_used += size;

#अगर_घोषित DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device, (अचिन्हित दीर्घ दीर्घ) size,
                      (अचिन्हित दीर्घ दीर्घ) ptr);
#पूर्ण_अगर
        वापस ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    व्योम मुक्त(व्योम * ptr, माप_प्रकार size) override {
#अगर_घोषित DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device, (अचिन्हित दीर्घ दीर्घ) size,
                      (अचिन्हित दीर्घ दीर्घ) ptr);
#पूर्ण_अगर

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (व्योम *) ((अक्षर *) pool_addr + pool_used));
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
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(पूर्णांक device) {
    std::string mem_pool_type = get_env_as_lowercase("GGML_CANN_MEM_POOL").value_or("");

    अगर (mem_pool_type == "prio") {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        वापस std::unique_ptr<ggml_cann_pool>(नया ggml_cann_pool_buf_prio(device));
    }

    अगर (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        वापस std::unique_ptr<ggml_cann_pool>(नया ggml_cann_pool_vmm(device));
    }

    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    वापस std::unique_ptr<ggml_cann_pool>(नया ggml_cann_pool_buf(device));
}

// cann buffer

/**
 * @brief Tracks multi-threaded write progress for a single tensor.
 *
 * When multiple threads call set_tensor on different chunks of the same tensor,
 * this tracker accumulates progress and defers post-processing (quantized format
 * transform or ND-to-NZ conversion) until all data has been written.
 */
काष्ठा TensorSetTracker {
    std::mutex mtx;                   ///< Protects concurrent access to this tracker
    माप_प्रकार bytes_written = 0;         ///< Accumulated bytes written so far
    माप_प्रकार योग_bytes = 0;           ///< Target size (full tensor)
    std::vector<uint8_t> host_बफर; ///< Host staging buffer for quantized tensors
};

/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
काष्ठा ggml_backend_cann_बफर_context {
    int32_t device;             ///< The device ID associated with this buffer context.
    व्योम *  dev_ptr = nullptr;  ///< Pointer to the device memory allocated for the buffer.

    std::mutex tracker_mutex;   ///< Protects the trackers map
    std::unordered_map<व्योम *, std::unique_ptr<TensorSetTracker>> trackers;

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_बफर_context(int32_t device, व्योम * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_बफर_context() { ACL_CHECK(aclrtFree(dev_ptr)); }

    /**
     * @brief Get or create a tracker for the given tensor.
     */
    TensorSetTracker * get_or_create_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        स्वतः कुंजी = tensor->तथ्य;
        स्वतः it = trackers.find(कुंजी);
        अगर (it == trackers.end()) {
            स्वतः tracker = std::make_unique<TensorSetTracker>();
            tracker->योग_bytes = ggml_nbytes(tensor);
            स्वतः * ptr = tracker.लो();
            trackers[कुंजी] = std::move(tracker);
            वापस ptr;
        }
        वापस it->second.लो();
    }

    /**
     * @brief Remove the tracker for the given tensor.
     */
    व्योम remove_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        trackers.erase(tensor->तथ्य);
    }
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
काष्ठा ggml_backend_cann_बफर_type_context {
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
अटल स्थिर अक्षर * ggml_backend_cann_बफर_type_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_cann_बफर_type_context * buft_ctx = (ggml_backend_cann_बफर_type_context *) buft->context;

    वापस buft_ctx->name.c_str();
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
अटल bool ggml_backend_buft_is_cann(ggml_backend_बफर_type_t buft) {
    वापस buft->iface.get_name == ggml_backend_cann_बफर_type_name;
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
अटल व्योम ggml_backend_cann_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_cann_बफर_context * ctx = (ggml_backend_cann_बफर_context *) बफर->context;
    मिटाओ ctx;
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
अटल व्योम * ggml_backend_cann_बफर_get_base(ggml_backend_बफर_t बफर) {
    ggml_backend_cann_बफर_context * ctx = (ggml_backend_cann_बफर_context *) बफर->context;
    वापस ctx->dev_ptr;
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
अटल व्योम ggml_backend_cann_transform_q4_0(ggml_tensor * tensor, स्थिर व्योम * src, व्योम * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    माप_प्रकार  quant_bytes = n_elems * माप(uint8_t) / 2;

    uint8_t *  quant_offset = (uint8_t *) dst;
    uint16_t * scale_offset = (uint16_t *) ((अक्षर *) dst + quant_bytes);

    क्रम (पूर्णांक i = 0; i < groups; i++) {
        स्थिर block_q4_0 * group = (स्थिर block_q4_0 *) ((स्थिर अक्षर *) src + i * माप(block_q4_0));
        *scale_offset            = group->d;
        scale_offset++;

        // 0-15
        क्रम (पूर्णांक j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] & 0x0F);
            (*quant_offset) |= ((group->qs[j + 1] << 4));
            quant_offset++;
        }

        // 16-31
        क्रम (पूर्णांक j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] >> 4);
            (*quant_offset) |= (group->qs[j + 1] & 0xF0);
            quant_offset++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    क्रम (quant_offset = (uint8_t *) dst; quant_offset < (uint8_t *) dst + quant_bytes; quant_offset++) {
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
अटल व्योम ggml_backend_cann_transform_back_q4_0(स्थिर ggml_tensor * tensor, व्योम * src, व्योम * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    माप_प्रकार  quant_bytes = n_elems * माप(uint8_t) / 2;

    uint8_t *  quant_offset = (uint8_t *) src;
    uint16_t * scale_offset = (uint16_t *) ((अक्षर *) src + quant_bytes);

    क्रम (; quant_offset < (uint8_t *) src + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
    quant_offset = (uint8_t *) src;

    क्रम (पूर्णांक i = 0; i < groups; i++) {
        block_q4_0 * group = (block_q4_0 *) ((अक्षर *) dst + i * माप(block_q4_0));
        group->d           = *scale_offset;
        scale_offset++;

        // 0-15
        क्रम (पूर्णांक j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j]     = ((*quant_offset) & 0x0F);
            group->qs[j + 1] = ((*quant_offset) >> 4);
            quant_offset++;
        }

        // 16-31
        क्रम (पूर्णांक j = 0; j < QK4_0 / 2; j += 2) {
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
अटल व्योम ggml_backend_cann_transform_q8_0(ggml_tensor * tensor, स्थिर व्योम * src, व्योम * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    माप_प्रकार  quant_bytes = n_elems * माप(uint8_t);

    uint8_t *  quant_offset = (uint8_t *) dst;
    uint16_t * scale_offset = (uint16_t *) ((अक्षर *) dst + quant_bytes);

    क्रम (पूर्णांक i = 0; i < groups; i++) {
        स्थिर block_q8_0 * group = (स्थिर block_q8_0 *) ((स्थिर अक्षर *) src + i * माप(block_q8_0));
        *scale_offset            = group->d;
        scale_offset++;
        माप_प्रकार group_quant_आकार = QK8_0 * माप(uint8_t);
        स_नकल(quant_offset, group->qs, group_quant_आकार);
        quant_offset += group_quant_आकार;
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
अटल व्योम ggml_backend_cann_transform_back_q8_0(स्थिर ggml_tensor * tensor, स्थिर व्योम * src, व्योम * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    माप_प्रकार  quant_bytes = n_elems * माप(uint8_t);

    स्थिर uint8_t *  quant_offset = (स्थिर uint8_t *) src;
    स्थिर uint16_t * scale_offset = (स्थिर uint16_t *) ((स्थिर अक्षर *) src + quant_bytes);

    क्रम (पूर्णांक i = 0; i < groups; i++) {
        block_q8_0 * group = (block_q8_0 *) ((अक्षर *) dst + i * माप(block_q8_0));
        group->d           = *scale_offset;
        scale_offset++;
        माप_प्रकार group_quant_आकार = QK8_0 * माप(uint8_t);
        स_नकल(group->qs, quant_offset, group_quant_आकार);
        quant_offset += group_quant_आकार;
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
अटल व्योम ggml_backend_cann_transform(ggml_tensor * tensor, स्थिर व्योम * src, व्योम * dst) {
    चयन (tensor->type) {
        हाल GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            अवरोध;
        हाल GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            अवरोध;
        शेष:
            अवरोध;
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
अटल व्योम ggml_backend_cann_transform_back(स्थिर ggml_tensor * tensor, व्योम * src, व्योम * dst) {
    चयन (tensor->type) {
        हाल GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_back_q4_0(tensor, src, dst);
            अवरोध;
        हाल GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_back_q8_0(tensor, src, dst);
            अवरोध;
        शेष:
            अवरोध;
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
अटल bool need_transform(ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q8_0:
            वापस true;
        शेष:
            वापस false;
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
अटल क्रमागत ggml_status ggml_backend_cann_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    अगर (tensor->view_src != शून्य && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->बफर->buft == बफर->buft);
        वापस GGML_STATUS_SUCCESS;
    }

    // TODO: cann backend doesn't support quantized yet. Just leave the code
    // here.
    अगर (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        माप_प्रकार original_आकार = ggml_nbytes(tensor);
        माप_प्रकार padded_आकार   = ggml_backend_buft_get_alloc_आकार(बफर->buft, tensor);

        अगर (padded_आकार > original_आकार && tensor->view_src == nullptr) {
            माप_प्रकार memset_आकार = padded_आकार - original_आकार;
            ACL_CHECK(aclrtMemset((अक्षर *) tensor->तथ्य + original_आकार, memset_आकार, 0, memset_आकार));
        }
    }
    वापस GGML_STATUS_SUCCESS;
}

/**
 * @brief Workspace for caching NZ buffers per device.
 *
 * This struct manages a device buffer used in NZ computations. It supports
 * allocation, reallocation, and clearing of cached memory. The struct is
 * designed to be used with a global array, one per device.
 */
काष्ठा ggml_cann_nz_workspace {
    std::mutex mtx;    // Protects ptr/allocated from concurrent access
    व्योम * ptr;        // Pointer to allocated device buffer
    माप_प्रकार allocated;  // Size of currently allocated buffer in bytes

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
    व्योम सफा() {
        अगर (ptr) {
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
    व्योम पुनः_स्मृति(माप_प्रकार new_आकार) {
        अगर (new_आकार > allocated) {
            सफा();
            ACL_CHECK(aclrtMalloc(&ptr, new_आकार, ACL_MEM_MALLOC_HUGE_FIRST));
            allocated = new_आकार;
        }
    }

    /**
     * @brief Get the device buffer pointer.
     *
     * @return Pointer to the allocated buffer, or nullptr if not allocated.
     */
    व्योम * लो() स्थिर { वापस ptr; }
};

/**
 * @brief Global array of NZ workspaces, one per device.
 */
अटल ggml_cann_nz_workspace g_nz_workspaces[GGML_CANN_MAX_DEVICES];

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
अटल व्योम weight_format_to_nz(ggml_tensor * tensor, पूर्णांक device) {
    acl_tensor_ptr weightTransposed = ggml_cann_create_tensor(tensor, tensor->ne, tensor->nb, 2, ACL_FORMAT_ND, 0);
    uint64_t       workspace_आकार    = 0;
    aclOpExecutor * executor;

    // TransMatmulWeight
    ACL_CHECK(aclnn_Trans_Matmul_Weight_Get_Workspace_आकार(weightTransposed.लो(), &workspace_आकार, &executor));

    std::lock_guard<std::mutex> lock(g_nz_workspaces[device].mtx);
    // Avoid frequent malloc/free of the workspace.
    g_nz_workspaces[device].पुनः_स्मृति(workspace_आकार);

    व्योम * g_nz_workspace = g_nz_workspaces[device].लो();

    ACL_CHECK(aclnnTransMatmulWeight(g_nz_workspace, workspace_आकार, executor, nullptr));
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
अटल व्योम ggml_backend_cann_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                                ggml_tensor *         tensor,
                                                स्थिर व्योम *          तथ्य,
                                                माप_प्रकार                offset,
                                                माप_प्रकार                size) {
    ggml_backend_cann_बफर_context * ctx = (ggml_backend_cann_बफर_context *) बफर->context;

    ggml_cann_set_device(ctx->device);

    // Only check env once.
    अटल bool weight_to_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    bool is_quantized = need_transform(tensor->type);
    bool is_nz        = !is_quantized && tensor->type != GGML_TYPE_BF16 && weight_to_nz &&
                 is_matmul_weight((स्थिर ggml_tensor *) tensor);

    // Plain tensor (not quantized, not NZ): direct copy, no tracking needed
    अगर (!is_quantized && !is_nz) {
        ACL_CHECK(aclrtMemcpy((अक्षर *) tensor->तथ्य + offset, size, तथ्य, size, ACL_MEMCPY_HOST_TO_DEVICE));
        वापस;
    }

    // Single-shot write (full tensor at once): handle directly without tracking overhead
    अगर (offset == 0 && size == ggml_nbytes(tensor)) {
        अगर (is_quantized) {
            व्योम * transform_बफर = दो_स्मृति(size);
            ggml_backend_cann_transform(tensor, तथ्य, transform_बफर);
            ACL_CHECK(aclrtMemcpy(tensor->तथ्य, size, transform_बफर, size, ACL_MEMCPY_HOST_TO_DEVICE));
            मुक्त(transform_बफर);
        } अन्यथा {
            // NZ weight
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ACL_CHECK(aclrtMemcpy(tensor->तथ्य, size, तथ्य, size, ACL_MEMCPY_HOST_TO_DEVICE));
            weight_format_to_nz(tensor, ctx->device);
        }
        वापस;
    }

    // Chunked write: use tracker to accumulate progress and defer transform/conversion
    TensorSetTracker * tracker = ctx->get_or_create_tracker(tensor);
    std::unique_lock<std::mutex> lock(tracker->mtx);

    अगर (is_quantized) {
        // Stage data in host buffer; transform requires full tensor data
        अगर (tracker->host_बफर.empty()) {
            tracker->host_बफर.resize(tracker->योग_bytes);
        }
        स_नकल(tracker->host_बफर.तथ्य() + offset, तथ्य, size);
    } अन्यथा {
        // NZ weight: upload chunk to device immediately, defer conversion
        ACL_CHECK(aclrtMemcpy((अक्षर *) tensor->तथ्य + offset, size, तथ्य, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    tracker->bytes_written += size;

    // All chunks received: perform deferred transform/conversion
    अगर (tracker->bytes_written >= tracker->योग_bytes) {
        अगर (is_quantized) {
            व्योम * transform_बफर = दो_स्मृति(tracker->योग_bytes);
            ggml_backend_cann_transform(tensor, tracker->host_बफर.तथ्य(), transform_बफर);
            ACL_CHECK(aclrtMemcpy(tensor->तथ्य, tracker->योग_bytes, transform_बफर, tracker->योग_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            मुक्त(transform_बफर);
        }

        अगर (is_nz) {
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
अटल व्योम ggml_backend_cann_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                स्थिर ggml_tensor *   tensor,
                                                व्योम *                तथ्य,
                                                माप_प्रकार                offset,
                                                माप_प्रकार                size) {
    ggml_backend_cann_बफर_context * ctx = (ggml_backend_cann_बफर_context *) बफर->context;

    ggml_cann_set_device(ctx->device);

    अगर (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(तथ्य, size, (अक्षर *) tensor->तथ्य + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
    } अन्यथा {
        व्योम * transform_बफर = दो_स्मृति(size);
        ACL_CHECK(aclrtMemcpy(transform_बफर, size, (अक्षर *) tensor->तथ्य + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
        ggml_backend_cann_transform_back(tensor, transform_बफर, तथ्य);
        मुक्त(transform_बफर);
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
अटल bool ggml_backend_cann_बफर_cpy_tensor(ggml_backend_बफर_t बफर,
                                                स्थिर ggml_tensor *   src,
                                                ggml_tensor *         dst) {
    अगर (ggml_backend_buft_is_cann(src->बफर->buft)) {
        ggml_backend_cann_बफर_context * src_ctx = (ggml_backend_cann_बफर_context *) src->बफर->context;
        ggml_backend_cann_बफर_context * dst_ctx = (ggml_backend_cann_बफर_context *) बफर->context;

        माप_प्रकार memcpy_आकार = ggml_nbytes(src);
        // Same device.
        अगर (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((अक्षर *) dst->तथ्य, memcpy_आकार, (स्थिर अक्षर *) src->तथ्य, memcpy_आकार,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE));
            वापस true;
        } अन्यथा {
#अगर_घोषित ASCEND_310P
            // TODO: Support 310p P2P copy
            वापस false;
#पूर्ण_अगर
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device, dst_ctx->device));
            अगर (canAccessPeer) {
                ggml_cann_set_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((अक्षर *) dst->तथ्य, memcpy_आकार, (स्थिर अक्षर *) src->तथ्य, memcpy_आकार,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE));
                वापस true;
            }
        }
    }
    वापस false;
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
अटल व्योम ggml_backend_cann_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_cann_बफर_context * ctx = (ggml_backend_cann_बफर_context *) बफर->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset((अक्षर *) tensor->तथ्य + offset, size, value, size));
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
अटल व्योम ggml_backend_cann_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_cann_बफर_context * ctx = (ggml_backend_cann_बफर_context *) बफर->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, बफर->size, value, बफर->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
अटल स्थिर ggml_backend_बफर_i ggml_backend_cann_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_cann_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_cann_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_cann_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cann_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cann_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_cann_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ ggml_backend_cann_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_बफर_clear,
    /* .reset           = */ शून्य,
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
अटल ggml_backend_बफर_t ggml_backend_cann_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    ggml_backend_cann_बफर_type_context * buft_ctx = (ggml_backend_cann_बफर_type_context *) buft->context;

    ggml_cann_set_device(buft_ctx->device);

    स्थिर माप_प्रकार alignment = 128;
    size                   = GGML_PAD(size, alignment);
    अगर (size == 0) {
        size = alignment;
    }
    व्योम *   dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    अगर (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, buft_ctx->device, acl_Get_Recent_Err_संदेश());
        वापस nullptr;
    }

    ggml_backend_cann_बफर_context * ctx = नया ggml_backend_cann_बफर_context(buft_ctx->device, dev_ptr);

    वापस ggml_backend_बफर_init(buft, ggml_backend_cann_बफर_interface, ctx, size);
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
अटल माप_प्रकार ggml_backend_cann_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस 128;

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
अटल माप_प्रकार ggml_backend_cann_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft,
                                                           स्थिर ggml_tensor *        tensor) {
    माप_प्रकार  size = ggml_nbytes(tensor);
    int64_t ne0  = tensor->ne[0];

    // Only check env once.
    अटल bool weight_to_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);
    अगर (ggml_is_quantized(tensor->type)) {
        अगर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    } अन्यथा अगर (weight_to_nz && tensor->type != GGML_TYPE_BF16
               && is_matmul_weight((स्थिर ggml_tensor *) tensor)) {
        // NZ format weight are not support quantized yet.
        // If ND tensor transform to NZ, size may changed.
        int64_t shape[] = { tensor->ne[1], tensor->ne[0] };
        GGML_ASSERT(tensor->ne[2] == 1);
        GGML_ASSERT(tensor->ne[3] == 1);
        स्थिर aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
        माप_प्रकार              new_आकार;
        ACL_CHECK(aclnn_Calculate_Matmul_Weight_आकार_V2(acl_shape, ggml_cann_type_mapping(tensor->type), &new_आकार));
        ACL_CHECK(aclDestroyIntArray(acl_shape));
        size = std::max(size, new_आकार);
    }

    वापस size;

    GGML_UNUSED(buft);
}

अटल bool ggml_backend_cann_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    वापस false;

    GGML_UNUSED(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
अटल स्थिर ggml_backend_बफर_type_i ggml_backend_cann_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_cann_बफर_type_name,
    /* .alloc_buffer        = */ ggml_backend_cann_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ शून्य,
    /* .get_alignment       = */ ggml_backend_cann_बफर_type_get_alignment,
    /* .get_max_size        = */ शून्य,  // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cann_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ ggml_backend_cann_बफर_type_is_host,
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
ggml_backend_बफर_type_t ggml_backend_cann_बफर_type(int32_t device) {
    अटल std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    अगर (device >= ggml_backend_cann_get_device_गिनती()) {
        वापस nullptr;
    }

    अटल ggml_backend_बफर_type ggml_backend_cann_बफर_types[GGML_CANN_MAX_DEVICES];

    अटल bool ggml_backend_cann_बफर_type_initialized = false;

    अगर (!ggml_backend_cann_बफर_type_initialized) {
        क्रम (int32_t i = 0; i < ggml_cann_info().device_गिनती; i++) {
            ggml_backend_cann_बफर_types[i] = {
                /* .iface    = */ ggml_backend_cann_बफर_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), i),
                /* .context  = */
                नया ggml_backend_cann_बफर_type_context{ i, "CANN" + std::to_string(i) },
            };
        }
        ggml_backend_cann_बफर_type_initialized = true;
    }

    वापस &ggml_backend_cann_बफर_types[device];
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
अटल स्थिर अक्षर * ggml_backend_cann_host_बफर_type_name(ggml_backend_बफर_type_t buft) {
    वापस "CANN_Host";

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
अटल स्थिर अक्षर * ggml_backend_cann_host_बफर_name(ggml_backend_बफर_t बफर) {
    वापस "CANN_Host";

    GGML_UNUSED(बफर);
}

/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer, including
 * its context.
 *
 * @param buffer The CANN host buffer to free.
 */
अटल व्योम ggml_backend_cann_host_बफर_free(ggml_backend_बफर_t बफर) {
    ACL_CHECK(aclrtFreeHost(बफर->context));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
अटल व्योम * ggml_cann_host_malloc(माप_प्रकार size) {
    अगर (दो_पर्या("GGML_CANN_NO_PINNED") != nullptr) {
        वापस nullptr;
    }

    स्थिर माप_प्रकार alignment = 128;
    size                   = GGML_PAD(size, alignment);
    अगर (size == 0) {
        size = alignment;
    }

    व्योम *   hostPtr = nullptr;
    aclError err     = aclrtMallocHost((व्योम **) &hostPtr, size);
    अगर (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                      acl_Get_Recent_Err_संदेश());
        वापस nullptr;
    }
    वापस hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
अटल ggml_backend_बफर_t ggml_backend_cann_host_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                                                             माप_प्रकार                     size) {
    व्योम * hostPtr = ggml_cann_host_malloc(size);

    अगर (hostPtr == nullptr) {
        // fallback to cpu buffer
        वापस ggml_backend_buft_alloc_बफर(ggml_backend_cpu_बफर_type(), size);
    }

    ggml_backend_बफर_t बफर = ggml_backend_cpu_बफर_from_ptr(hostPtr, size);
    बफर->buft                 = buft;
    बफर->iface.free_बफर    = ggml_backend_cann_host_बफर_free;

    वापस बफर;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_बफर_type_t ggml_backend_cann_host_बफर_type() {
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_cann_बफर_type_host = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cann_host_बफर_type_name,
                           /* .alloc_buffer         = */ ggml_backend_cann_host_बफर_type_alloc_बफर,
                           /* .alloc_buffer_n       = */ शून्य,
                           /* .get_alignment        = */ ggml_backend_cpu_बफर_type()->iface.get_alignment,
                           /* .get_max_size         = */ शून्य,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ ggml_backend_cpu_बफर_type()->iface.get_alloc_आकार,
                           /* .get_alloc_size_n     = */ शून्य,
                           /* .is_host              = */ ggml_backend_cpu_बफर_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
        /* .context  = */ nullptr,
    };

    वापस &ggml_backend_cann_बफर_type_host;
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
अटल bool ggml_cann_compute_forward(ggml_backend_cann_context & ctx, काष्ठा ggml_tensor * dst) {
    चयन (dst->op) {
        हाल GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            अवरोध;
        हाल GGML_OP_GET_ROWS:
            ggml_cann_get_rows(ctx, dst);
            अवरोध;
        हाल GGML_OP_SET_ROWS:
            ggml_cann_set_rows(ctx, dst);
            अवरोध;
        हाल GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            अवरोध;
        हाल GGML_OP_ADD:
        हाल GGML_OP_ADD1:
            ggml_cann_द्विआधारी_op<aclnn_add>(ctx, dst);
            अवरोध;
        हाल GGML_OP_SUB:
            ggml_cann_द्विआधारी_op<aclnn_sub>(ctx, dst);
            अवरोध;
        हाल GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            अवरोध;
        हाल GGML_OP_MUL:
            ggml_cann_द्विआधारी_op<aclnn_mul>(ctx, dst);
            अवरोध;
        हाल GGML_OP_DIV:
            ggml_cann_द्विआधारी_op<aclnn_div>(ctx, dst);
            अवरोध;
        हाल GGML_OP_UNARY:
            चयन (ggml_get_unary_op(dst)) {
                हाल GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_OP_UNARY(Abs);
                    अवरोध;
                हाल GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_OP_UNARY(Neg);
                    अवरोध;
                हाल GGML_UNARY_OP_GELU:
                हाल GGML_UNARY_OP_GELU_ERF:
                    // aclnnGelu internally uses the erf-based approximation.
                    GGML_CANN_CALL_OP_UNARY(Gelu);
                    अवरोध;
                हाल GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_OP_UNARY(Silu);
                    अवरोध;
                हाल GGML_UNARY_OP_GELU_QUICK:
                    {
                        स्वतः lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                        };
                        ggml_cann_op_unary(lambda, ctx, dst);
                    }
                    अवरोध;
                हाल GGML_UNARY_OP_TANH:
                    GGML_CANN_CALL_OP_UNARY(Tanh);
                    अवरोध;
                हाल GGML_UNARY_OP_RELU:
                    GGML_CANN_CALL_OP_UNARY(Relu);
                    अवरोध;
                हाल GGML_UNARY_OP_SIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Sigmoid);
                    अवरोध;
                हाल GGML_UNARY_OP_HARDSIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Hardsigmoid);
                    अवरोध;
                हाल GGML_UNARY_OP_HARDSWISH:
                    GGML_CANN_CALL_OP_UNARY(Hardswish);
                    अवरोध;
                हाल GGML_UNARY_OP_EXP:
                    GGML_CANN_CALL_OP_UNARY(Exp);
                    अवरोध;
                हाल GGML_UNARY_OP_ELU:
                    ggml_cann_elu(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_SGN:
                    GGML_CANN_CALL_OP_UNARY(Sign);
                    अवरोध;
                हाल GGML_UNARY_OP_STEP:
                    ggml_cann_step(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_SOFTPLUS:
                    ggml_cann_softplus(ctx, dst);
                    अवरोध;
                शेष:
                    वापस false;
            }
            अवरोध;
        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(dst)) {
                हाल GGML_GLU_OP_REGLU:
                    GGML_CANN_CALL_OP_UNARY_GATED(Relu);
                    अवरोध;
                हाल GGML_GLU_OP_GEGLU:
                    ggml_cann_geglu(ctx, dst, 0);  // approximate=0 → tanh
                    अवरोध;
                हाल GGML_GLU_OP_GEGLU_ERF:
                    ggml_cann_geglu(ctx, dst, 1);  // approximate=1 → erf
                    अवरोध;
                हाल GGML_GLU_OP_SWIGLU:
                    ggml_cann_swiglu(ctx, dst);
                    अवरोध;
                हाल GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cann_swiglu_clamp(ctx, dst);
                    अवरोध;
                हाल GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cann_geglu_quick(ctx, dst);
                    अवरोध;
                शेष:
                    वापस false;
            }
            अवरोध;
        हाल GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            अवरोध;
        हाल GGML_OP_GROUP_NORM:
            ggml_cann_group_norm(ctx, dst);
            अवरोध;
        हाल GGML_OP_L2_NORM:
            ggml_cann_l2_norm(ctx, dst);
            अवरोध;
        हाल GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cann_cross_entropy_loss(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            अवरोध;
        हाल GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            अवरोध;
        हाल GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            अवरोध;
        हाल GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            अवरोध;
        हाल GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            अवरोध;
        हाल GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            अवरोध;
        हाल GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            अवरोध;
        हाल GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            अवरोध;
        हाल GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            अवरोध;
        हाल GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            अवरोध;
        हाल GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == nullptr);
            dst->src[1] = dst->src[0];
            ggml_cann_द्विआधारी_op<aclnn_mul>(ctx, dst);
            अवरोध;
        हाल GGML_OP_SQRT:
            GGML_CANN_CALL_OP_UNARY(Sqrt);
            अवरोध;
        हाल GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            अवरोध;
        हाल GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            अवरोध;
        हाल GGML_OP_SET:
            ggml_cann_set(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            अवरोध;
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
            अवरोध;
        हाल GGML_OP_DIAG_MASK_INF:
            ggml_cann_diag_mask(ctx, dst, -अनन्त);
            अवरोध;
        हाल GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            अवरोध;
        हाल GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            अवरोध;
        हाल GGML_OP_IM2COL:
            ggml_cann_im2col(ctx, dst);
            अवरोध;
        हाल GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            अवरोध;
        हाल GGML_OP_योग:
            ggml_cann_योग(ctx, dst);
            अवरोध;
        हाल GGML_OP_योग_ROWS:
            ggml_cann_योग_rows(ctx, dst);
            अवरोध;
        हाल GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            अवरोध;
        हाल GGML_OP_ARGMAX:
            ggml_cann_argmax(ctx, dst);
            अवरोध;
        हाल GGML_OP_COS:
            ggml_cann_op_unary<aclnn_cos>(ctx, dst);
            अवरोध;
        हाल GGML_OP_SIN:
            ggml_cann_op_unary<aclnn_sin>(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cann_conv_transpose_1d(ctx, dst);
            अवरोध;
        हाल GGML_OP_LOG:
            GGML_CANN_CALL_OP_UNARY(Log);
            अवरोध;
        हाल GGML_OP_MEAN:
            ggml_cann_mean(ctx, dst);
            अवरोध;
        हाल GGML_OP_PAD_REFLECT_1D:
            ggml_cann_pad_reflect_1d(ctx, dst);
            अवरोध;
        हाल GGML_OP_गिनती_EQUAL:
            ggml_cann_गिनती_equal(ctx, dst);
            अवरोध;
        हाल GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            अवरोध;
        हाल GGML_OP_OUT_PROD:
            ggml_cann_out_prod(ctx, dst);
            अवरोध;
        हाल GGML_OP_GATED_LINEAR_ATTN:
            ggml_cann_gated_linear_attn(ctx, dst);
            अवरोध;
        हाल GGML_OP_SSM_CONV:
            ggml_cann_ssm_conv(ctx, dst);
            अवरोध;
        हाल GGML_OP_CUMSUM:
            ggml_cann_cumsum(ctx, dst);
            अवरोध;
        हाल GGML_OP_TRI:
            ggml_cann_tri(ctx, dst);
            अवरोध;
        हाल GGML_OP_FILL:
            ggml_cann_fill(ctx, dst);
            अवरोध;
        हाल GGML_OP_DIAG:
            ggml_cann_diag(ctx, dst);
            अवरोध;
        हाल GGML_OP_SOLVE_TRI:
            ggml_cann_solve_tri(ctx, dst);
            अवरोध;
        शेष:
            वापस false;
    }

    वापस true;
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
अटल स्थिर अक्षर * ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;

    वापस cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
अटल व्योम ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    मिटाओ cann_ctx;
    मिटाओ backend;
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
अटल व्योम ggml_backend_cann_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *  tensor,
                                               स्थिर व्योम *   तथ्य,
                                               माप_प्रकार         offset,
                                               माप_प्रकार         size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_बफर_t       buf      = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;

    GGML_ASSERT(buf->buft == ggml_backend_cann_बफर_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync((अक्षर *) tensor->तथ्य + offset, size, तथ्य, size, ACL_MEMCPY_HOST_TO_DEVICE,
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
अटल व्योम ggml_backend_cann_get_tensor_async(ggml_backend_t      backend,
                                               स्थिर ggml_tensor * tensor,
                                               व्योम *              तथ्य,
                                               माप_प्रकार              offset,
                                               माप_प्रकार              size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_बफर_t       buf      = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;

    GGML_ASSERT(buf->buft == ggml_backend_cann_बफर_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync(तथ्य, size, (अक्षर *) tensor->तथ्य + offset, size, ACL_MEMCPY_DEVICE_TO_HOST,
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
अटल bool ggml_backend_cann_cpy_tensor_async(ggml_backend_t      backend_src,
                                               ggml_backend_t      backend_dst,
                                               स्थिर ggml_tensor * src,
                                               ggml_tensor *       dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) || ggml_backend_is_cann(backend_dst));

    GGML_ASSERT(!is_matmul_weight((स्थिर ggml_tensor *) src));

    अगर (!ggml_backend_buft_is_cann(src->बफर->buft) || !ggml_backend_buft_is_cann(dst->बफर->buft)) {
        वापस false;
    }

    ggml_backend_बफर_t buf_src = src->view_src ? src->view_src->बफर : src->बफर;
    ggml_backend_बफर_t buf_dst = dst->view_src ? dst->view_src->बफर : dst->बफर;

    ggml_backend_cann_context * cann_ctx_src = (ggml_backend_cann_context *) backend_src->context;
    ggml_backend_cann_context * cann_ctx_dst = (ggml_backend_cann_context *) backend_dst->context;

    माप_प्रकार copy_आकार = ggml_nbytes(dst);
    अगर (copy_आकार == 0) {
        वापस true;
    }
    अगर (backend_src != backend_dst) {
#अगर_घोषित ASCEND_310P
        // TODO: Support 310p P2P copy
        वापस false;
#पूर्ण_अगर
        ggml_backend_cann_बफर_context * buf_ctx_src = (ggml_backend_cann_बफर_context *) buf_src->context;
        ggml_backend_cann_बफर_context * buf_ctx_dst = (ggml_backend_cann_बफर_context *) buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device, cann_ctx_dst->device));
        अगर (!canAccessPeer) {
            वापस false;
        }

        // need open both directions for memcpyasync between devices.
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_set_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        // wait for task_queue empty to keep task order.
        ACL_CHECK(aclrtMemcpyAsync(dst->तथ्य, copy_आकार, src->तथ्य, copy_आकार, ACL_MEMCPY_DEVICE_TO_DEVICE,
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
    } अन्यथा {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->तथ्य, copy_आकार, src->तथ्य, copy_आकार, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    वापस true;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
अटल व्योम ggml_backend_cann_synchronize(ggml_backend_t backend) {
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
अटल bool ggml_cann_can_fuse(स्थिर काष्ठा ggml_cgraph *          cgraph,
                               पूर्णांक                                 गाँठ_idx,
                               std::initializer_list<क्रमागत ggml_op> ops) {
    अगर (!ggml_can_fuse(cgraph, गाँठ_idx, ops)) {
        वापस false;
    }

    // CANN backend supports fusing ADD + RMS_NORM operations
    अगर ((ops.size() == 2) && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_RMS_NORM) {
        ggml_tensor * add_गाँठ = cgraph->nodes[गाँठ_idx];
        // TODO: support broadcast for ADD + RMS_NORM
        अगर (add_गाँठ->src[0]->ne[0] != add_गाँठ->src[1]->ne[0] || add_गाँठ->src[0]->ne[1] != add_गाँठ->src[1]->ne[1] ||
            add_गाँठ->src[0]->ne[2] != add_गाँठ->src[1]->ne[2] || add_गाँठ->src[0]->ne[3] != add_गाँठ->src[1]->ne[3]) {
            वापस false;
        }
        वापस true;
    }

    वापस false;
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
अटल व्योम evaluate_and_capture_cann_graph(ggml_backend_cann_context * cann_ctx,
                                            ggml_cgraph *               cgraph,
                                            bool                        use_cann_graph,
                                            bool                        cann_graph_capture_required) {
#अगर_घोषित USE_ACL_GRAPH
    अगर (use_cann_graph && cann_graph_capture_required) {  // Begin CANN graph capture
        ACL_CHECK(aclmdlRICaptureBegin(cann_ctx->stream(), ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    }
#पूर्ण_अगर  // USE_ACL_GRAPH
    // Only perform the graph execution if CANN graphs are not enabled, or we are capturing the graph.
    // With the use of CANN graphs, the execution will be performed by the graph launch.
    अटल bool opt_fusion = parse_bool(get_env_as_lowercase("GGML_CANN_OPERATOR_FUSION").value_or(""));

    अगर (!use_cann_graph || cann_graph_capture_required) {
        क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * गाँठ = cgraph->nodes[i];
            अगर (opt_fusion) {
                अगर (ggml_cann_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM })) {
                    ggml_cann_op_add_rms_norm_fused(*cann_ctx, गाँठ, cgraph->nodes[i + 1]);
                    i++;
                    जारी;
                }
            }

            अगर (ggml_is_empty(गाँठ) || गाँठ->op == GGML_OP_RESHAPE || गाँठ->op == GGML_OP_TRANSPOSE ||
                गाँठ->op == GGML_OP_VIEW || गाँठ->op == GGML_OP_PERMUTE || गाँठ->op == GGML_OP_NONE) {
                जारी;
            }

            अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                जारी;
            }

            bool ok = ggml_cann_compute_forward(*cann_ctx, गाँठ);
            अगर (!ok) {
                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, गाँठ->name, ggml_op_name(गाँठ->op));
            }
            GGML_ASSERT(ok);
        }
    }

#अगर_घोषित USE_ACL_GRAPH
    अगर (use_cann_graph) {
        GGML_ASSERT(!cann_ctx->graph_lru_cache.cache_list.empty());
        ggml_cann_graph * matched_graph = cann_ctx->graph_lru_cache.cache_list.front();

        अगर (cann_graph_capture_required) {  // End CANN graph capture
            ACL_CHECK(aclmdlRICaptureEnd(cann_ctx->stream(), &matched_graph->graph));
        }

        // Execute CANN graph
        ACL_CHECK(aclmdlRIExecuteAsync(matched_graph->graph, cann_ctx->stream()));
    }
#पूर्ण_अगर  // USE_ACL_GRAPH
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
अटल क्रमागत ggml_status ggml_backend_cann_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_set_device(cann_ctx->device);
    g_nz_workspaces[cann_ctx->device].सफा();

    // calculate rope cache for fist layer in current device.
    cann_ctx->rope_cache.cached = false;

    bool graph_capture_required = false;
#अगर_घोषित USE_ACL_GRAPH
    bool use_cann_graph = true;

    अटल bool prefill_use_graph = parse_bool(get_env_as_lowercase("GGML_CANN_PREFILL_USE_GRAPH").value_or(""));
    अगर (!prefill_use_graph) {
        // Do not use acl_graph for prefill.
        क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * गाँठ = cgraph->nodes[i];
            // TODO: Optimize here. Currently, we can only
            // get seq_len by FA's input.
            अगर (गाँठ->op == GGML_OP_FLASH_ATTN_EXT) {
                // Q -> src[0], shape: [B, S, N, D]
                use_cann_graph = (गाँठ->src[0]->ne[1] == 1);
                अवरोध;
            }
        }
    }

    अगर (!cann_ctx->acl_graph_mode) {
        use_cann_graph = false;
    }

    अगर (use_cann_graph) {
        // If no matching graph is found, the graph needs to be recaptured.
        graph_capture_required = !cann_ctx->graph_lru_cache.find_and_move_to_front(cgraph);

        अगर (graph_capture_required) {
            // If no matching graph is found, add a new ACL graph.
            ggml_cann_graph * new_graph = ggml_cann_graph::create_from_cgraph(cgraph);
            cann_ctx->graph_lru_cache.धकेलो(new_graph);

            // Pre-load rope cache before graph capture.  During capture the
            // stream cannot perform host-to-device memcpy or device memory
            // malloc/free.  Running the full cache init now populates the
            // cache metadata so these branches are skipped during capture,
            // while also warming up the memory pool.
            क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * गाँठ = cgraph->nodes[i];
                अगर (गाँठ->op == GGML_OP_ROPE) {
                    ggml_cann_rope_cache_preload(*cann_ctx, गाँठ);
                    अवरोध;
                }
            }
        }
    }
#अन्यथा
    bool use_cann_graph = false;
#पूर्ण_अगर  // USE_ACL_GRAPH
    evaluate_and_capture_cann_graph(cann_ctx, cgraph, use_cann_graph, graph_capture_required);

    वापस GGML_STATUS_SUCCESS;
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
अटल bool ggml_backend_cann_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    चयन (op->op) {
        हाल GGML_OP_UNARY:
            चयन (ggml_get_unary_op(op)) {
                हाल GGML_UNARY_OP_ABS:
                हाल GGML_UNARY_OP_NEG:
                हाल GGML_UNARY_OP_GELU:
                हाल GGML_UNARY_OP_SILU:
                हाल GGML_UNARY_OP_RELU:
                हाल GGML_UNARY_OP_SIGMOID:
                हाल GGML_UNARY_OP_HARDSIGMOID:
                हाल GGML_UNARY_OP_HARDSWISH:
                हाल GGML_UNARY_OP_GELU_QUICK:
                हाल GGML_UNARY_OP_TANH:
                हाल GGML_UNARY_OP_EXP:
                हाल GGML_UNARY_OP_ELU:
                हाल GGML_UNARY_OP_SGN:
                हाल GGML_UNARY_OP_STEP:
                हाल GGML_UNARY_OP_GELU_ERF:
                हाल GGML_UNARY_OP_SOFTPLUS:
                    वापस true;
                शेष:
                    वापस false;
            }
        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(op)) {
                हाल GGML_GLU_OP_REGLU:
                हाल GGML_GLU_OP_GEGLU:
                हाल GGML_GLU_OP_SWIGLU:
                हाल GGML_GLU_OP_GEGLU_ERF:
                हाल GGML_GLU_OP_GEGLU_QUICK:
                हाल GGML_GLU_OP_SWIGLU_CLAMP:
                    वापस true;
                शेष:
                    वापस false;
            }
            अवरोध;
        हाल GGML_OP_MUL_MAT:
            {
                चयन (op->src[0]->type) {
#अगर_अघोषित ASCEND_310P
                    हाल GGML_TYPE_BF16:
#पूर्ण_अगर
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_F32:
                        वापस true;
                    हाल GGML_TYPE_Q8_0:
                    हाल GGML_TYPE_Q4_0:
#अगर_घोषित ASCEND_310P
                        // Q4 && Q8 per group is not support on 310p device
                        वापस false;
#अन्यथा
                        // only support contiguous for quantized types.
                        वापस ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#पूर्ण_अगर
                    शेष:
                        वापस false;
                }
            }
        हाल GGML_OP_MUL_MAT_ID:
            चयन (op->src[0]->type) {
                हाल GGML_TYPE_F16:
                हाल GGML_TYPE_F32:
                    वापस true;
                हाल GGML_TYPE_Q8_0:
                हाल GGML_TYPE_Q4_0:
#अगर_घोषित ASCEND_310P
                    // Q4 && Q8 per group is not support on 310p device
                    वापस false;
#अन्यथा
                    // only support contiguous for quantized types.
                    वापस ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
#पूर्ण_अगर
                शेष:
                    वापस false;
            }
        // embedding
        हाल GGML_OP_GET_ROWS:
            {
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
#अगर_अघोषित ASCEND_310P
                    हाल GGML_TYPE_BF16:
#पूर्ण_अगर
                    हाल GGML_TYPE_Q8_0:
                        वापस true;
                    शेष:
                        वापस false;
                }
            }
            अवरोध;
        हाल GGML_OP_SET_ROWS:
            {
                चयन (op->type) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
#अगर_अघोषित ASCEND_310P
                    हाल GGML_TYPE_BF16:
#पूर्ण_अगर
                        वापस true;
                    शेष:
                        वापस false;
                }
            }
            अवरोध;
        हाल GGML_OP_CPY:
            {
                ggml_tensor * src = op->src[0];
#अगर_घोषित ASCEND_310P
                अगर ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                    // only support F32 and F16 on 310P.
                    वापस false;
                }
#अन्यथा
                अगर ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
                    // only support F32, F16 and BF16.
                    वापस false;
                }
#पूर्ण_अगर
                वापस true;
            }
            अवरोध;
        हाल GGML_OP_CONT:
            {
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_F16:
#अगर_अघोषित ASCEND_310P
                    हाल GGML_TYPE_BF16:
#पूर्ण_अगर
                        वापस true;
                    शेष:
                        वापस false;
                }
            }
        हाल GGML_OP_ROPE:
            {
                अगर (((स्थिर int32_t *) op->op_params)[15] != 0) {
                    वापस false; // FIXME: support ggml_rope_set_offset
                }
                अगर (op->src[0]->ne[0] > 896) {
                    वापस false;
                }
#अगर_घोषित ASCEND_310P
                // TODO: Support rope_dim < ne00(dim)
                अगर (op->src[0]->ne[0] != op->op_params[1]) {
                    वापस false;
                }
                अगर (!ggml_is_contiguous(op->src[0])) {
                    वापस false;
                }
#पूर्ण_अगर
                वापस true;
            }
        हाल GGML_OP_UPSCALE:
            {
                // aclnnUpsampleNearest2dGetWorkspaceSize not support
                // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
                अगर (op->src[0]->ne[2] * op->ne[3] != op->src[0]->ne[3] * op->ne[2]) {
                    वापस false;
                }
                अगर (op->op_params[0] != GGML_SCALE_MODE_NEAREST) {
                    वापस false;
                }
                अगर (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                    वापस false;
                }
                वापस true;
            }
        हाल GGML_OP_POOL_2D:
            {
                स्थिर int32_t * opts = (स्थिर int32_t *) op->op_params;
#अगर_घोषित ASCEND_310P
                क्रमागत ggml_op_pool opt = static_cast<ggml_op_pool>(opts[0]);
                अगर (opt == GGML_OP_POOL_MAX) {
                    वापस false;
                }
#पूर्ण_अगर
                स्थिर पूर्णांक k0 = opts[1];
                स्थिर पूर्णांक k1 = opts[2];
                स्थिर पूर्णांक p0 = opts[5];
                स्थिर पूर्णांक p1 = opts[6];
                // value of paddingH should be at most half of kernelH
                // value of paddingW should be at most half of kernelW
                वापस (p0 <= (k0 / 2)) && (p1 <= (k1 / 2));
            }
        हाल GGML_OP_योग:
            वापस ggml_is_contiguous_rows(op->src[0]);
        हाल GGML_OP_L2_NORM:
        हाल GGML_OP_CROSS_ENTROPY_LOSS:
        हाल GGML_OP_DUP:
        हाल GGML_OP_IM2COL:
        हाल GGML_OP_CONCAT:
        हाल GGML_OP_REPEAT:
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_NORM:
        हाल GGML_OP_ADD:
        हाल GGML_OP_ADD1:
        हाल GGML_OP_SUB:
        हाल GGML_OP_MUL:
        हाल GGML_OP_DIV:
        हाल GGML_OP_RMS_NORM:
        हाल GGML_OP_SQR:
        हाल GGML_OP_SQRT:
        हाल GGML_OP_CLAMP:
        हाल GGML_OP_DIAG_MASK_INF:
        हाल GGML_OP_योग_ROWS:
        हाल GGML_OP_ARGSORT:
        हाल GGML_OP_ACC:
        हाल GGML_OP_SET:
        हाल GGML_OP_GROUP_NORM:
            वापस true;
        हाल GGML_OP_PAD:
            // TODO: add circular padding support for cann, see https://github.com/ggml-org/llama.cpp/pull/16985
            वापस ggml_get_op_params_i32(op, 8) == 0;
        हाल GGML_OP_ARANGE:
        हाल GGML_OP_TIMESTEP_EMBEDDING:
        हाल GGML_OP_LEAKY_RELU:
        हाल GGML_OP_ARGMAX:
        हाल GGML_OP_COS:
        हाल GGML_OP_SIN:
        हाल GGML_OP_LOG:
        हाल GGML_OP_MEAN:
        हाल GGML_OP_PAD_REFLECT_1D:
        हाल GGML_OP_गिनती_EQUAL:
        हाल GGML_OP_GATED_LINEAR_ATTN:
            वापस true;
        हाल GGML_OP_OUT_PROD:
            {
#अगर_घोषित ASCEND_310P
                // Ger is not supported on 310p device
                वापस false;
#अन्यथा
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_F32:
                        वापस true;
                    शेष:
                        वापस false;
                }
#पूर्ण_अगर
            }
        हाल GGML_OP_CONV_TRANSPOSE_1D:
            वापस true;
        हाल GGML_OP_SCALE:
            भग्न bias;
            स_नकल(&bias, (स्थिर भग्न *) (op->op_params) + 1, माप(भग्न));
            वापस bias == 0.0f;  // TODO: support bias != 0.0f
        हाल GGML_OP_SOFT_MAX:
            // TODO: support attention sinks [TAG_ATTN_SINKS]
            अगर (op->src[2]) {
                वापस false;
            }
            वापस true;
        हाल GGML_OP_FLASH_ATTN_EXT:
            {
#अगर_घोषित ASCEND_310P
                // FA not support on 310p device
                वापस false;
#अन्यथा
                // derived from [ggml-cuda.cu]
                अगर (op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16) {
                    वापस false;
                }
                अगर (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32 &&
                    op->src[1]->type != GGML_TYPE_BF16) {
                    वापस false;
                }
                अगर (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16) {
                    वापस false;
                }
                // TODO: support attention sinks [TAG_ATTN_SINKS]
                अगर (op->src[4]) {
                    वापस false;
                }
                अगर (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                    // different head sizes of K and V are not supported yet
                    वापस false;
                }
                भग्न logitSoftcap = 0.0f;
                स_नकल(&logitSoftcap, (स्थिर भग्न *) (op->op_params) + 2, माप(भग्न));
                अगर (logitSoftcap != 0.0f) {
                    वापस false;
                }
                वापस true;
#पूर्ण_अगर
            }
        हाल GGML_OP_SSM_CONV:
            वापस true;
        हाल GGML_OP_CUMSUM:
            वापस op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_TRI:
            वापस op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_FILL:
            वापस op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_DIAG:
            वापस op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_SOLVE_TRI:
            वापस op->src[0]->type == GGML_TYPE_F32;
        शेष:
            वापस false;
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
अटल व्योम ggml_backend_cann_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
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
अटल व्योम ggml_backend_cann_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    अगर (ggml_backend_is_cann(backend)) {
        ACL_CHECK(aclrtStreamWaitEvent(cann_ctx->stream(), (aclrtEvent) event->context));
    } अन्यथा {
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
अटल स्थिर ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_cann_graph_compute,
    /* .event_record            = */ ggml_backend_cann_event_record,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .graph_optimize          = */ शून्य,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
अटल ggml_guid_t ggml_backend_cann_guid() {
    अटल ggml_guid guid = { 0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                              0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64 };
    वापस &guid;
}

// backend device
काष्ठा ggml_backend_cann_device_context {
    पूर्णांक         device;
    std::string name;
    std::string description;
    पूर्णांक op_offload_min_batch_आकार;
};

अटल स्थिर अक्षर * ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    वापस ctx->name.c_str();
}

अटल स्थिर अक्षर * ggml_backend_cann_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    वापस ctx->description.c_str();
}

अटल व्योम ggml_backend_cann_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ggml_backend_cann_get_device_memory(ctx->device, मुक्त, योग);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_cann_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    वापस GGML_BACKEND_DEVICE_TYPE_GPU;
}

अटल व्योम ggml_backend_cann_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type        = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_memory(dev, &props->memory_free, &props->memory_योग);

    bool host_बफर = दो_पर्या("GGML_CANN_NO_PINNED") == nullptr;

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ host_बफर,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
    };
}

अटल ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    GGML_UNUSED(params);
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    वापस ggml_backend_cann_init(ctx->device);
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
अटल bool ggml_backend_cann_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    अगर (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context *      dev_ctx  = (ggml_backend_cann_device_context *) dev->context;
        ggml_backend_cann_बफर_type_context * buft_ctx = (ggml_backend_cann_बफर_type_context *) buft->context;
        वापस buft_ctx->device == dev_ctx->device;
    }
    वापस false;
}

अटल ggml_backend_बफर_type_t ggml_backend_cann_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    वापस ggml_backend_cann_बफर_type(ctx->device);
}

अटल ggml_backend_बफर_type_t ggml_backend_cann_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    वापस ggml_backend_cann_host_बफर_type();
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
अटल bool ggml_backend_cann_offload_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;

    वापस op->ne[1] >= dev_ctx->op_offload_min_batch_आकार && op->op != GGML_OP_GET_ROWS;
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
अटल ggml_backend_event_t ggml_backend_cann_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *) dev->context;

    ggml_cann_set_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    वापस नया ggml_backend_event{
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
अटल व्योम ggml_backend_cann_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent) event->context));

    मिटाओ event;
    GGML_UNUSED(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
अटल व्योम ggml_backend_cann_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent) event->context));

    GGML_UNUSED(dev);
}

अटल स्थिर ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_memory,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,  // called for every card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_बफर_type,
    /* .get_host_buffer_type    = */ ggml_backend_cann_device_get_host_बफर_type,
    /* .buffer_from_host_ptr    = */ शून्य,  // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};

// backend reg
काष्ठा ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

अटल स्थिर अक्षर * ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    वापस GGML_CANN_NAME;
}

अटल माप_प्रकार ggml_backend_cann_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    वापस ctx->devices.size();
}

अटल ggml_backend_dev_t ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    GGML_ASSERT(सूचकांक < ctx->devices.size());
    वापस ctx->devices[सूचकांक];
}

अटल व्योम * ggml_backend_cann_reg_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    // reserved for future use
    वापस nullptr;
}

अटल स्थिर ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_गिनती,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_address,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    अटल ggml_backend_reg reg;
    अटल bool             initialized = false;

    {
        अटल std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        अगर (!initialized) {
            aclInit(nullptr);
            ggml_backend_cann_reg_context * ctx = नया ggml_backend_cann_reg_context;
            स्थिर पूर्णांक min_batch_आकार = दो_पर्या("GGML_OP_OFFLOAD_MIN_BATCH") ? म_से_प(दो_पर्या("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            क्रम (पूर्णांक i = 0; i < ggml_cann_info().device_गिनती; i++) {
                ggml_backend_cann_device_context * dev_ctx = नया ggml_backend_cann_device_context();
                dev_ctx->description                       = aclrtGetSocName();
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_CANN_NAME + std::to_string(i);
                dev_ctx->op_offload_min_batch_आकार         = min_batch_आकार;
                ggml_cann_set_device(i);
                ggml_backend_dev_t dev = नया ggml_backend_device{ /* .iface   = */ ggml_backend_cann_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_cann_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = true;
    }

    वापस &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(nullptr);
    अगर (device < 0 || device >= ggml_backend_cann_get_device_गिनती()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        वापस nullptr;
    }

    ggml_backend_cann_context * ctx = नया ggml_backend_cann_context(device);
    अगर (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        वापस nullptr;
    }
    ggml_cann_set_device(ctx->device);
    ggml_backend_t cann_backend =
        नया ggml_backend{ /* .guid      = */ ggml_backend_cann_guid(),
                          /* .interface = */ ggml_backend_cann_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                          /* .context   = */ ctx };

    वापस cann_backend;
}

bool ggml_backend_is_cann(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_गिनती() {
    वापस ggml_cann_info().device_गिनती;
}

व्योम ggml_backend_cann_get_device_description(int32_t device, अक्षर * description, माप_प्रकार description_आकार) {
    ggml_cann_set_device(device);
    स्थिर अक्षर * soc_name = aclrtGetSocName();
    snprintf(description, description_आकार, "%s", soc_name);
}

व्योम ggml_backend_cann_get_device_memory(int32_t device, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    ggml_cann_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, मुक्त, योग));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)

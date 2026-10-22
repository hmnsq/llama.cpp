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

#جيڪڏهن_وصف_نه_ٿيل CANN_COMMON_H
#وصف_ڪريو CANN_COMMON_H

#ضمن "../ggml-impl.h"
#ضمن "../include/ggml-cann.h"
#ضمن "../include/ggml.h"

#ضمن <acl/acl.h>
#ضمن <unistd.h>

#ضمن <atomic>
#ضمن <condition_variable>
#ضمن <cstdio>
#ضمن <functional>
#ضمن <iostream>
#ضمن <list>
#ضمن <map>
#ضمن <memory>
#ضمن <mutex>
#ضمن <optional>
#ضمن <string>
#ضمن <thread>
#ضمن <vector>

#وصف_ڪريو ميٽرڪس_صف_PADDING    512
#وصف_ڪريو GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] فراغ ggml_cann_error(ثابت حرف * stmt, ثابت حرف * func, ثابت حرف * file, صحيح سٽ, ثابت حرف * نياپو);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#وصف_ڪريو ACL_CHECK_GEN(stmt, success, error_fn)                                \
    افعل {                                                                      \
        صحيح err_ڪوڊ = (stmt);                                                \
        إذا (err_ڪوڊ != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } طالما (0);

#وصف_ڪريو ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, acl_Get_Recent_Err_نياپو)

/**
 * @brief Contains information about CANN devices.
 */
بنية ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_ڳڻپ;

    /**
     * @brief Information about a single CANN device.
     */
    بنية cann_device_info {
        صحيح    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t ڪل_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

ثابت ggml_cann_device_info & ggml_cann_info();

فراغ    ggml_cann_سيٽ_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(ثابت std::string & name);
bool                       parse_bool(ثابت std::string & قدر);
صحيح                        parse_integer(ثابت std::string & قدر);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
بنية ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    مجازي ~ggml_cann_pool() = افتراضي;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    مجازي فراغ * alloc(size_t سائيز, size_t * actual_سائيز) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    مجازي فراغ حرر(فراغ * ptr, size_t سائيز) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
بنية ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    فراغ *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_سائيز = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = افتراضي;

    /**
     * @brief Constructor that initializes the memory pool.
     * @param pool Reference to the memory pool.
     */
    explicit ggml_cann_pool_alloc(ggml_cann_pool & pool) : pool(&pool) {}

    /**
     * @brief Constructor that initializes the memory pool and allocates memory.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     */
    ggml_cann_pool_alloc(ggml_cann_pool & pool, size_t سائيز) : pool(&pool) { alloc(سائيز); }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        إذا (ptr != nullptr) {
            pool->حرر(ptr, actual_سائيز);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    فراغ * alloc(size_t سائيز) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(سائيز, &this->actual_سائيز);
        أرجع ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    فراغ * alloc(ggml_cann_pool & pool, size_t سائيز) {
        this->pool = &pool;
        أرجع alloc(سائيز);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    فراغ * get() { أرجع ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(ثابت ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & عامل=(ثابت ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & عامل=(ggml_cann_pool_alloc &&) = delete;
};

#جيڪڏهن_وصف_ٿيل USE_ACL_گراف
بنية ggml_گراف_ڳنڍ_properties {
    // dst tensor
    فراغ *    ڳنڍ_پتو;
    ggml_type ڳنڍ_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    فراغ *    src_پتو[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op ڳنڍ_op;
    int32_t op_params[GGML_MAX_OP_PARAMS / حجم(int32_t)];

    /**
     * @brief Check if a ggml tensor node matches this property set.
     *
     * This function compares all relevant fields (address, op type, shape, source inputs, op params)
     * to determine whether the current node matches these previously recorded properties.
     *
     * @param node The current ggml tensor node.
     * @return true if all fields match (excluding GGML_OP_VIEW); false otherwise.
     */
    bool has_matching_properties(ggml_tensor * ڳنڍ) {
        إذا (ڳنڍ->معلومات != this->ڳنڍ_پتو && ڳنڍ->op != GGML_OP_VIEW) {
            أرجع false;
        }

        إذا (ڳنڍ->op != this->ڳنڍ_op) {
            أرجع false;
        }

        إذا (ڳنڍ->type != this->ڳنڍ_type) {
            أرجع false;
        }

        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            إذا (ڳنڍ->ne[i] != this->ne[i]) {
                أرجع false;
            }
            إذا (ڳنڍ->nb[i] != this->nb[i]) {
                أرجع false;
            }
        }

        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            إذا (ڳنڍ->src[i]) {
                إذا (ڳنڍ->src[i]->معلومات != this->src_پتو[i] && ڳنڍ->op != GGML_OP_VIEW) {
                    أرجع false;
                }

                إذا (ڳنڍ->src[i]->type != this->src_type[i]) {
                    أرجع false;
                }

                لكل (صحيح d = 0; d < GGML_MAX_DIMS; d++) {
                    إذا (ڳنڍ->src[i]->ne[d] != this->src_ne[i][d]) {
                        أرجع false;
                    }
                    إذا (ڳنڍ->src[i]->nb[d] != this->src_nb[i][d]) {
                        أرجع false;
                    }
                }
            } وإلا {
                إذا (this->src_پتو[i] != nullptr) {
                    أرجع false;
                }
            }
        }

        أرجع memcmp(this->op_params, ڳنڍ->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

بنية ggml_cann_گراف {
    ~ggml_cann_گراف() {
        إذا (گراف != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(گراف));
        }
    }

    aclmdlRI گراف = nullptr;

    std::ويڪٽر<ggml_گراف_ڳنڍ_properties> ggml_گراف_properties;

    /**
     * @brief Create a new CANN graph from a ggml computation graph.
     *
     * This function creates a new ggml_cann_graph object and fills its node properties
     * (operation type, dimensions, strides, input sources, and operation parameters)
     * based on the current ggml computation graph.
     *
     * Each node in the ggml graph is mapped to a property entry in the new CANN graph:
     * - node address
     * - operation type
     * - shape (ne) and strides (nb)
     * - source tensor addresses
     * - operation parameters
     *
     * @param cgraph The current ggml computation graph.
     * @return Pointer to the newly created ggml_cann_graph object.
     */
    ساكن ggml_cann_گراف * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_گراف * new_گراف = new ggml_cann_گراف();
        new_گراف->ggml_گراف_properties.resize(cgraph->n_nodes);

        لكل (صحيح ڳنڍ_idx = 0; ڳنڍ_idx < cgraph->n_nodes; ++ڳنڍ_idx) {
            ggml_tensor * ڳنڍ = cgraph->nodes[ڳنڍ_idx];
            خودڪار &        prop = new_گراف->ggml_گراف_properties[ڳنڍ_idx];

            prop.ڳنڍ_پتو = ڳنڍ->معلومات;
            prop.ڳنڍ_op      = ڳنڍ->op;
            prop.ڳنڍ_type    = ڳنڍ->type;

            std::copy_n(ڳنڍ->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(ڳنڍ->nb, GGML_MAX_DIMS, prop.nb);

            لكل (صحيح src = 0; src < GGML_MAX_SRC; ++src) {
                إذا (ڳنڍ->src[src]) {
                    prop.src_پتو[src] = ڳنڍ->src[src]->معلومات;
                    prop.src_type[src]    = ڳنڍ->src[src]->type;
                    std::copy_n(ڳنڍ->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(ڳنڍ->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } وإلا {
                    prop.src_پتو[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_ڳڻپ;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            يادگيري_نقل(prop.op_params, ڳنڍ->op_params, GGML_MAX_OP_PARAMS);
        }

        أرجع new_گراف;
    }

    /**
     * @brief Check whether this CANN graph matches the given ggml computation graph.
     *
     * This function compares the number of nodes and each node's properties
     * (operation type, dimensions, strides, inputs, and operation parameters)
     * to determine whether this CANN graph matches the given ggml graph.
     *
     * @param cgraph The current ggml computation graph.
     * @return true if this CANN graph matches the ggml graph; false otherwise.
     */
    bool matches_cgraph(ggml_cgraph * cgraph) {
        إذا (this->ggml_گراف_properties.سائيز() != static_cast<size_t>(cgraph->n_nodes)) {
            أرجع false;
        }

        لكل (صحيح i = 0; i < cgraph->n_nodes; ++i) {
            إذا (!this->ggml_گراف_properties[i].has_matching_properties(cgraph->nodes[i])) {
                أرجع false;
            }
        }

        أرجع true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
بنية ggml_cann_گراف_lru_cache {
    size_t صلاحيت;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_گراف *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_گراف_lru_cache() { صلاحيت = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").قدر_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    فراغ ڌڪو(ggml_cann_گراف * new_ڳنڍ) {
        إذا (cache_list.سائيز() >= صلاحيت) {
            ggml_cann_گراف * old = cache_list.پٺ();
            cache_list.ڪڍو_پٺ();
            delete old;  // free the old graph
        }
        cache_list.ڌڪو_front(new_ڳنڍ);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    فراغ clear() {
        لكل (خودڪار ptr : cache_list) {
            delete ptr;
        }
        cache_list.clear();
    }

    /**
     * @brief Destructor that clears the cache and frees all cached graphs.
     */
    ~ggml_cann_گراف_lru_cache() { clear(); }

    /**
     * @brief Find a cached CANN graph that matches the given ggml graph and move it to front.
     *
     * This function iterates through the cached CANN graphs stored in the LRU cache and
     * compares them against the given ggml computation graph. If a matching graph is found,
     * it is promoted to the front of the LRU cache and returned. Otherwise, the function
     * returns nullptr.
     *
     * @param cgraph The current ggml computation graph.
     * @return true if found; false otherwise.
     */
    bool ڳوليو_and_move_منزل_front(ggml_cgraph * cgraph) {
        لكل (خودڪار & گراف_ptr : this->cache_list) {
            إذا (گراف_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(گراف_ptr);
                cache_list.ڌڪو_front(گراف_ptr);
                أرجع true;
            }
        }
        أرجع false;
    }
};
#جيڪڏهن_پورو  // USE_ACL_GRAPH

بنية ggml_cann_rope_cache {
    ~ggml_cann_rope_cache() {
        إذا (theta_scale_cache) {
            ACL_CHECK(aclrtFree(theta_scale_cache));
        }
        إذا (sin_cache) {
            ACL_CHECK(aclrtFree(sin_cache));
        }
        إذا (cos_cache) {
            ACL_CHECK(aclrtFree(cos_cache));
        }
        إذا (position_select_اشاريو) {
            ACL_CHECK(aclrtFree(position_select_اشاريو));
        }
        إذا (theta_scale_exp_host) {
            حرر(theta_scale_exp_host);
        }
        إذا (position_select_اشاريو_host) {
            حرر(position_select_اشاريو_host);
        }
        إذا (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_ڊيگهائي,
               int64_t position_ڊيگهائي,
               عائم   ext_ضرب_ڪندڙ,
               عائم   theta_scale,
               عائم   freq_scale,
               عائم   attn_ضرب_ڪندڙ,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_استعمال_ٿيل,
               bool    is_imrope,
               صحيح     sections[4]) {
        أرجع this->theta_scale_ڊيگهائي == theta_scale_ڊيگهائي && this->position_ڊيگهائي == position_ڊيگهائي &&
               this->ext_ضرب_ڪندڙ == ext_ضرب_ڪندڙ && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_ضرب_ڪندڙ == attn_ضرب_ڪندڙ && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_استعمال_ٿيل == mrope_استعمال_ٿيل && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    فراغ سيٽ(int64_t theta_scale_ڊيگهائي,
             int64_t position_ڊيگهائي,
             عائم   ext_ضرب_ڪندڙ,
             عائم   theta_scale,
             عائم   freq_scale,
             عائم   attn_ضرب_ڪندڙ,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_استعمال_ٿيل,
             bool    is_imrope,
             صحيح     sections[4]) {
        this->theta_scale_ڊيگهائي = theta_scale_ڊيگهائي;
        this->position_ڊيگهائي    = position_ڊيگهائي;
        this->ext_ضرب_ڪندڙ         = ext_ضرب_ڪندڙ;
        this->theta_scale        = theta_scale;
        this->freq_scale         = freq_scale;
        this->attn_ضرب_ڪندڙ        = attn_ضرب_ڪندڙ;
        this->is_neox            = is_neox;
        this->indep_sects        = indep_sects;
        this->mrope_استعمال_ٿيل         = mrope_استعمال_ٿيل;
        this->is_imrope          = is_imrope;
        this->sections[0]        = sections[0];
        this->sections[1]        = sections[1];
        this->sections[2]        = sections[2];
        this->sections[3]        = sections[3];
    }

    // memory cache, prepare before inferencing.
    فراغ *  theta_scale_cache          = nullptr;
    عائم * theta_scale_exp_host       = nullptr;
    صحيح *   position_select_اشاريو_host = nullptr;
    فراغ *  position_select_اشاريو      = nullptr;
    فراغ *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    فراغ *  sin_cache                  = nullptr;
    فراغ *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_ڊيگهائي         = 0;
    int64_t position_ڊيگهائي            = 0;
    bool    cached                     = false;
    عائم   ext_ضرب_ڪندڙ                 = 0.0f;
    عائم   theta_scale                = 0.0f;
    عائم   freq_scale                 = 0.0f;
    عائم   attn_ضرب_ڪندڙ                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_استعمال_ٿيل                 = false;
    صحيح     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

بنية ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        إذا (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    فراغ *  cache = nullptr;
    int64_t سائيز  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
بنية ggml_backend_cann_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    aclrtEvent  copy_event = nullptr; /**< Event for managing copy operations. */
#جيڪڏهن_وصف_ٿيل USE_ACL_گراف
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    ggml_cann_گراف_lru_cache گراف_lru_cache;
    bool                      acl_گراف_گهڻائي = true;
#جيڪڏهن_پورو
    bool                   async_گهڻائي;
    // Rope Cache
    ggml_cann_rope_cache   rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrt_وهڪرو streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(صحيح device) : device(device), name("CANN" + std::منزل_string(device)) {
        ggml_cann_سيٽ_device(device);
        description = aclrtGetSocName();

#جيڪڏهن_وصف_ٿيل USE_ACL_گراف
        acl_گراف_گهڻائي = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").قدر_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_گراف_گهڻائي ? "GRAPH" : "EAGER",
                      acl_گراف_گهڻائي ? "acl graph enabled" : "acl graph disabled");
#جيڪڏهن_پورو
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_سيٽ_device(device);
        إذا (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        لكل (صحيح i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            إذا (streams[i] != nullptr) {
                ACL_CHECK(aclrt_Destroy_وهڪرو(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrt_وهڪرو وهڪرو(صحيح وهڪرو) {
        إذا (streams[وهڪرو] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrt_سيٽ_Device(device));
            ACL_CHECK(aclrt_Create_وهڪرو(&streams[وهڪرو]));
        }
        أرجع streams[وهڪرو];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrt_وهڪرو وهڪرو() { أرجع وهڪرو(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    ساكن std::unique_ptr<ggml_cann_pool> new_pool_for_device(صحيح device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        إذا (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        أرجع *mem_pool;
    }
};

#جيڪڏهن_پورو  // CANN_COMMON_H

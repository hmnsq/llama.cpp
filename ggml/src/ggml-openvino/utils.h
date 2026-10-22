#ضمن "ggml-decoder.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-openvino-extra.h"

#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <functional>
#ضمن <memory>
#ضمن <mutex>
#ضمن <openvino/runtime/core.hpp>
#ضمن <openvino/runtime/infer_request.hpp>
#ضمن <string>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <utility>
#ضمن <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
بنية گراف_ڪنجي {
    صحيح n_nodes;
    صحيح n_leaves;
    std::string پهريون_ڳنڍ_name;
    std::string آخري_ڳنڍ_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::ويڪٽر<std::string> داخلا_srcs;

    گراف_ڪنجي(ثابت ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        إذا (n_nodes > 0) {
            پهريون_ڳنڍ_name = cgraph->nodes[0]->name;
            آخري_ڳنڍ_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ساكن ثابت bool ڀريل_ڪنجي = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        إذا (!ڀريل_ڪنجي && !include_inputs) {
            أرجع;
        }

        std::unordered_سيٽ<ثابت ggml_tensor *> ڳنڍ_سيٽ;
        ڳنڍ_سيٽ.reserve(cgraph->n_nodes);
        لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
            ڳنڍ_سيٽ.وجهو(cgraph->nodes[i]);
        }

        لكل (صحيح ڳنڍ_idx = 0; ڳنڍ_idx < cgraph->n_nodes; ڳنڍ_idx++) {
            ثابت ggml_tensor * ڳنڍ = cgraph->nodes[ڳنڍ_idx];
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ثابت ggml_tensor * src = ڳنڍ->src[src_idx];
                إذا (src == nullptr || src->بفر == nullptr || ڳنڍ_سيٽ.ڳڻپ(src) || src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                    استمر;
                }

                داخلا_srcs.ڌڪو_پٺ(std::منزل_string(ڳنڍ_idx) + ":" + std::منزل_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(ڳنڍ));
            }
        }
    }

    bool عامل==(ثابت گراف_ڪنجي & other) ثابت {
        أرجع n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               پهريون_ڳنڍ_name == other.پهريون_ڳنڍ_name &&
               آخري_ڳنڍ_name == other.آخري_ڳنڍ_name && داخلا_srcs == other.داخلا_srcs;
    }
};

بنية گراف_ڪنجي_hash {
    size_t عامل()(ثابت گراف_ڪنجي & ڪنجي) ثابت {
        size_t hash = std::hash<صحيح>{}(ڪنجي.n_nodes);
        hash ^= std::hash<صحيح>{}(ڪنجي.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        إذا (ڪنجي.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(ڪنجي.پهريون_ڳنڍ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(ڪنجي.آخري_ڳنڍ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        لكل (ثابت خودڪار & s : ڪنجي.داخلا_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        أرجع hash;
    }
};

بنية decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

بنية ov_compiled_گراف {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::ويڪٽر<std::string> داخلا_نالا;
    std::ويڪٽر<std::string> output_نالا;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
بنية ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_گراف> graphs;
    size_t backend_ڳڻپ = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
بنية ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<گراف_ڪنجي, std::shared_ptr<decoder_runtime_ctx>, گراف_ڪنجي_hash> decoder_cache;
    std::unordered_map<گراف_ڪنجي, std::shared_ptr<ov::InferRequest>, گراف_ڪنجي_hash> infer_request_cache;
    std::unordered_map<گراف_ڪنجي, std::shared_ptr<ov::InferRequest>, گراف_ڪنجي_hash> infer_request_cache_prefill;
    std::unordered_map<گراف_ڪنجي, std::ويڪٽر<std::string>, گراف_ڪنجي_hash> ov_داخلا_نالا_cache;
    std::unordered_map<گراف_ڪنجي, std::ويڪٽر<std::string>, گراف_ڪنجي_hash> ov_output_نالا_cache;
    size_t stateful_kv_سائيز;
    std::map<std::string, std::string> kv_state_داخلا_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    بنية compute_گراف {
        ggml_cgraph گراف;
        std::ويڪٽر<ggml_tensor *> nodes;
    };
    std::unordered_map<ثابت ggml_cgraph *, compute_گراف> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_سائيز(0) {}

    فراغ clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_داخلا_نالا_cache.clear();
        ov_output_نالا_cache.clear();
        kv_state_داخلا_name_map.clear();
        stateful_kv_سائيز = 0;
    }

    فراغ clear_caches() {
        std::تالو_guard<std::mutex> تالو(ctx_mutex);
        clear_caches_locked();
    }
};

تعداد ggml_حالت ov_گراف_compute(بنية ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ثابت فراغ * معلومات, size_t سائيز);

bool save_ggml_tensor_معلومات_منزل_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_رستو);

فراغ print_داخلا_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor);

فراغ print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت فراغ * output_dst);

سانچو <typename T>
std::ويڪٽر<T> pad_داخلا(ثابت T * معلومات,
                         size_t صفون,
                         size_t کانن_ڳڻپ_2,
                         size_t padded_صفون,
                         size_t padded_کانن_ڳڻپ,
                         T pad_قدر) {
    std::ويڪٽر<T> padded(padded_صفون * padded_کانن_ڳڻپ, pad_قدر);

    لكل (size_t i = 0; i < std::min(صفون, padded_صفون); ++i) {
        لكل (size_t j = 0; j < std::min(کانن_ڳڻپ_2, padded_کانن_ڳڻپ); ++j) {
            padded[i * padded_کانن_ڳڻپ + j] = معلومات[i * کانن_ڳڻپ_2 + j];
        }
    }

    أرجع padded;
}

سانچو <typename T>
std::ويڪٽر<T> pad_داخلا(ثابت ggml_tensor * tensor, size_t padded_صفون, size_t padded_کانن_ڳڻپ, T pad_قدر) {
    أرجع pad_داخلا<T>(reinterpret_cast<ثابت T *>(tensor->معلومات),
                        static_cast<size_t>(tensor->ne[1]),  //صفون
                        static_cast<size_t>(tensor->ne[0]),  //کانن_ڳڻپ
                        padded_صفون, padded_کانن_ڳڻپ, pad_قدر);
}

ثابت ggml_tensor * get_inp_pos_tensor(بنية ggml_cgraph * cgraph);

int64_t get_inp_pos_n_ٽوڪن(بنية ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool get_is_prefill(بنية ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool is_naive(بنية ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(بنية ggml_cgraph * cgraph);

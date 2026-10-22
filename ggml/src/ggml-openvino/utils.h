#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-decoder.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino-extra.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <functional>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/core.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/runtime/infer_request.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_set>
#ਅਨ੍ਤਰ੍ਭਾਵ <utility>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
ਬਣਤਰ ਗ੍ਰਾਫ_ਕੁੰਜੀ {
    ਪੂਰਨ n_nodes;
    ਪੂਰਨ n_leaves;
    std::string ਪਹਿਲਾ_ਗੰਢ_name;
    std::string ਆਖਰੀ_ਗੰਢ_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::ਸਦਿਸ਼<std::string> ਨਿਵੇਸ਼_srcs;

    ਗ੍ਰਾਫ_ਕੁੰਜੀ(ਅਚਲ ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        ਜੇ (n_nodes > 0) {
            ਪਹਿਲਾ_ਗੰਢ_name = cgraph->nodes[0]->name;
            ਆਖਰੀ_ਗੰਢ_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ਸਥਿਰ ਅਚਲ bool ਭਰਿਆ_ਕੁੰਜੀ = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        ਜੇ (!ਭਰਿਆ_ਕੁੰਜੀ && !include_inputs) {
            ਵਾਪਸ;
        }

        std::unordered_ਸੈੱਟ<ਅਚਲ ggml_tensor *> ਗੰਢ_ਸੈੱਟ;
        ਗੰਢ_ਸੈੱਟ.reserve(cgraph->n_nodes);
        ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
            ਗੰਢ_ਸੈੱਟ.ਪਾਓ(cgraph->nodes[i]);
        }

        ਲਈ (ਪੂਰਨ ਗੰਢ_idx = 0; ਗੰਢ_idx < cgraph->n_nodes; ਗੰਢ_idx++) {
            ਅਚਲ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];
            ਲਈ (ਪੂਰਨ src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ਅਚਲ ggml_tensor * src = ਗੰਢ->src[src_idx];
                ਜੇ (src == nullptr || src->ਬਫਰ == nullptr || ਗੰਢ_ਸੈੱਟ.ਗਿਣਤੀ_2(src) || src->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS) {
                    ਜਾਰੀ;
                }

                ਨਿਵੇਸ਼_srcs.ਧੱਕੋ_ਪਿੱਠ(std::ਮੰਜ਼ਿਲ_string(ਗੰਢ_idx) + ":" + std::ਮੰਜ਼ਿਲ_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(ਗੰਢ));
            }
        }
    }

    bool ਸੰਚਾਲਕ==(ਅਚਲ ਗ੍ਰਾਫ_ਕੁੰਜੀ & other) ਅਚਲ {
        ਵਾਪਸ n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               ਪਹਿਲਾ_ਗੰਢ_name == other.ਪਹਿਲਾ_ਗੰਢ_name &&
               ਆਖਰੀ_ਗੰਢ_name == other.ਆਖਰੀ_ਗੰਢ_name && ਨਿਵੇਸ਼_srcs == other.ਨਿਵੇਸ਼_srcs;
    }
};

ਬਣਤਰ ਗ੍ਰਾਫ_ਕੁੰਜੀ_hash {
    size_t ਸੰਚਾਲਕ()(ਅਚਲ ਗ੍ਰਾਫ_ਕੁੰਜੀ & ਕੁੰਜੀ) ਅਚਲ {
        size_t hash = std::hash<ਪੂਰਨ>{}(ਕੁੰਜੀ.n_nodes);
        hash ^= std::hash<ਪੂਰਨ>{}(ਕੁੰਜੀ.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        ਜੇ (ਕੁੰਜੀ.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(ਕੁੰਜੀ.ਪਹਿਲਾ_ਗੰਢ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(ਕੁੰਜੀ.ਆਖਰੀ_ਗੰਢ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & s : ਕੁੰਜੀ.ਨਿਵੇਸ਼_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        ਵਾਪਸ hash;
    }
};

ਬਣਤਰ decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

ਬਣਤਰ ov_compiled_ਗ੍ਰਾਫ {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::ਸਦਿਸ਼<std::string> ਨਿਵੇਸ਼_ਨਾਂ;
    std::ਸਦਿਸ਼<std::string> output_ਨਾਂ;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
ਬਣਤਰ ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_ਗ੍ਰਾਫ> graphs;
    size_t backend_ਗਿਣਤੀ = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
ਬਣਤਰ ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<ਗ੍ਰਾਫ_ਕੁੰਜੀ, std::shared_ptr<decoder_runtime_ctx>, ਗ੍ਰਾਫ_ਕੁੰਜੀ_hash> decoder_cache;
    std::unordered_map<ਗ੍ਰਾਫ_ਕੁੰਜੀ, std::shared_ptr<ov::InferRequest>, ਗ੍ਰਾਫ_ਕੁੰਜੀ_hash> infer_request_cache;
    std::unordered_map<ਗ੍ਰਾਫ_ਕੁੰਜੀ, std::shared_ptr<ov::InferRequest>, ਗ੍ਰਾਫ_ਕੁੰਜੀ_hash> infer_request_cache_prefill;
    std::unordered_map<ਗ੍ਰਾਫ_ਕੁੰਜੀ, std::ਸਦਿਸ਼<std::string>, ਗ੍ਰਾਫ_ਕੁੰਜੀ_hash> ov_ਨਿਵੇਸ਼_ਨਾਂ_cache;
    std::unordered_map<ਗ੍ਰਾਫ_ਕੁੰਜੀ, std::ਸਦਿਸ਼<std::string>, ਗ੍ਰਾਫ_ਕੁੰਜੀ_hash> ov_output_ਨਾਂ_cache;
    size_t stateful_kv_ਮਾਪ;
    std::map<std::string, std::string> kv_state_ਨਿਵੇਸ਼_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    ਬਣਤਰ compute_ਗ੍ਰਾਫ {
        ggml_cgraph ਗ੍ਰਾਫ;
        std::ਸਦਿਸ਼<ggml_tensor *> nodes;
    };
    std::unordered_map<ਅਚਲ ggml_cgraph *, compute_ਗ੍ਰਾਫ> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_ਮਾਪ(0) {}

    ਖਾਲੀ clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_ਨਿਵੇਸ਼_ਨਾਂ_cache.clear();
        ov_output_ਨਾਂ_cache.clear();
        kv_state_ਨਿਵੇਸ਼_name_map.clear();
        stateful_kv_ਮਾਪ = 0;
    }

    ਖਾਲੀ clear_caches() {
        std::ਤਾਲਾ_guard<std::mutex> ਤਾਲਾ(ctx_mutex);
        clear_caches_locked();
    }
};

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ov_ਗ੍ਰਾਫ_compute(ਬਣਤਰ ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ);

bool save_ggml_tensor_ਅੰਕੜੇ_ਮੰਜ਼ਿਲ_txt(ਅਚਲ ggml_tensor * tensor, ਅਚਲ std::string & file_ਰਸਤਾ);

ਖਾਲੀ print_ਨਿਵੇਸ਼_tensor_info(ਅਚਲ std::string & name, ਅਚਲ ov::Tensor & tensor);

ਖਾਲੀ print_output_tensor_info(ਅਚਲ std::string & name, ਅਚਲ ov::Tensor & tensor, ਅਚਲ ਖਾਲੀ * output_dst);

ਸਾਂਚਾ <typename T>
std::ਸਦਿਸ਼<T> pad_ਨਿਵੇਸ਼(ਅਚਲ T * ਅੰਕੜੇ,
                         size_t ਪੰਕਤੀਆਂ,
                         size_t ਖਾਨੇ_ਗਿਣਤੀ_2,
                         size_t padded_ਪੰਕਤੀਆਂ,
                         size_t padded_ਖਾਨੇ_ਗਿਣਤੀ,
                         T pad_ਮੁੱਲ) {
    std::ਸਦਿਸ਼<T> padded(padded_ਪੰਕਤੀਆਂ * padded_ਖਾਨੇ_ਗਿਣਤੀ, pad_ਮੁੱਲ);

    ਲਈ (size_t i = 0; i < std::min(ਪੰਕਤੀਆਂ, padded_ਪੰਕਤੀਆਂ); ++i) {
        ਲਈ (size_t j = 0; j < std::min(ਖਾਨੇ_ਗਿਣਤੀ_2, padded_ਖਾਨੇ_ਗਿਣਤੀ); ++j) {
            padded[i * padded_ਖਾਨੇ_ਗਿਣਤੀ + j] = ਅੰਕੜੇ[i * ਖਾਨੇ_ਗਿਣਤੀ_2 + j];
        }
    }

    ਵਾਪਸ padded;
}

ਸਾਂਚਾ <typename T>
std::ਸਦਿਸ਼<T> pad_ਨਿਵੇਸ਼(ਅਚਲ ggml_tensor * tensor, size_t padded_ਪੰਕਤੀਆਂ, size_t padded_ਖਾਨੇ_ਗਿਣਤੀ, T pad_ਮੁੱਲ) {
    ਵਾਪਸ pad_ਨਿਵੇਸ਼<T>(reinterpret_cast<ਅਚਲ T *>(tensor->ਅੰਕੜੇ),
                        static_cast<size_t>(tensor->ne[1]),  //ਪੰਕਤੀਆਂ
                        static_cast<size_t>(tensor->ne[0]),  //ਖਾਨੇ_ਗਿਣਤੀ
                        padded_ਪੰਕਤੀਆਂ, padded_ਖਾਨੇ_ਗਿਣਤੀ, pad_ਮੁੱਲ);
}

ਅਚਲ ggml_tensor * get_inp_pos_tensor(ਬਣਤਰ ggml_cgraph * cgraph);

int64_t get_inp_pos_n_ਟੋਕਨ(ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ggml_tensor * inp_pos);

bool get_is_prefill(ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ggml_tensor * inp_pos);

bool is_naive(ਬਣਤਰ ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(ਬਣਤਰ ggml_cgraph * cgraph);

#ಅನ್ತರ್ಭಾವ "ggml-decoder.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-openvino-extra.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <functional>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/core.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/runtime/infer_request.hpp>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <utility>
#ಅನ್ತರ್ಭಾವ <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
ರಚನೆ ನಕ್ಷೆ_ಕೀಲಿ {
    ಪೂರ್ಣಾಂಕ n_nodes;
    ಪೂರ್ಣಾಂಕ n_leaves;
    std::string ಮೊದಲ_ಗಂಟು_name;
    std::string ಕೊನೆಯ_ಗಂಟು_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::ಸದಿಶ<std::string> ಒಳಹರಿವು_srcs;

    ನಕ್ಷೆ_ಕೀಲಿ(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        ಆದರೆ (n_nodes > 0) {
            ಮೊದಲ_ಗಂಟು_name = cgraph->nodes[0]->name;
            ಕೊನೆಯ_ಗಂಟು_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ bool ತುಂಬಿದ_ಕೀಲಿ = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        ಆದರೆ (!ತುಂಬಿದ_ಕೀಲಿ && !include_inputs) {
            ಹಿಂತಿರುಗಿ;
        }

        std::unordered_ಗಣ<ಸ್ಥಿರಾಂಕ ggml_tensor *> ಗಂಟು_ಗಣ;
        ಗಂಟು_ಗಣ.reserve(cgraph->n_nodes);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
            ಗಂಟು_ಗಣ.ಸೇರಿಸು(cgraph->nodes[i]);
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_idx = 0; ಗಂಟು_idx < cgraph->n_nodes; ಗಂಟು_idx++) {
            ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src = ಗಂಟು->src[src_idx];
                ಆದರೆ (src == nullptr || src->ಬಫರ್ == nullptr || ಗಂಟು_ಗಣ.ಎಣಿಕೆ_2(src) || src->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS) {
                    ಮುಂದುವರಿ;
                }

                ಒಳಹರಿವು_srcs.ತಳ್ಳು_ಹಿಂಭಾಗ(std::ಗಮ್ಯ_string(ಗಂಟು_idx) + ":" + std::ಗಮ್ಯ_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(ಗಂಟು));
            }
        }
    }

    bool ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ ನಕ್ಷೆ_ಕೀಲಿ & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               ಮೊದಲ_ಗಂಟು_name == other.ಮೊದಲ_ಗಂಟು_name &&
               ಕೊನೆಯ_ಗಂಟು_name == other.ಕೊನೆಯ_ಗಂಟು_name && ಒಳಹರಿವು_srcs == other.ಒಳಹರಿವು_srcs;
    }
};

ರಚನೆ ನಕ್ಷೆ_ಕೀಲಿ_hash {
    size_t ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ನಕ್ಷೆ_ಕೀಲಿ & ಕೀಲಿ) ಸ್ಥಿರಾಂಕ {
        size_t hash = std::hash<ಪೂರ್ಣಾಂಕ>{}(ಕೀಲಿ.n_nodes);
        hash ^= std::hash<ಪೂರ್ಣಾಂಕ>{}(ಕೀಲಿ.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        ಆದರೆ (ಕೀಲಿ.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(ಕೀಲಿ.ಮೊದಲ_ಗಂಟು_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(ಕೀಲಿ.ಕೊನೆಯ_ಗಂಟು_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & s : ಕೀಲಿ.ಒಳಹರಿವು_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        ಹಿಂತಿರುಗಿ hash;
    }
};

ರಚನೆ decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

ರಚನೆ ov_compiled_ನಕ್ಷೆ {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::ಸದಿಶ<std::string> ಒಳಹರಿವು_ಹೆಸರುಗಳು;
    std::ಸದಿಶ<std::string> output_ಹೆಸರುಗಳು;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
ರಚನೆ ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_ನಕ್ಷೆ> graphs;
    size_t backend_ಎಣಿಕೆ = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
ರಚನೆ ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<ನಕ್ಷೆ_ಕೀಲಿ, std::shared_ptr<decoder_runtime_ctx>, ನಕ್ಷೆ_ಕೀಲಿ_hash> decoder_cache;
    std::unordered_map<ನಕ್ಷೆ_ಕೀಲಿ, std::shared_ptr<ov::InferRequest>, ನಕ್ಷೆ_ಕೀಲಿ_hash> infer_request_cache;
    std::unordered_map<ನಕ್ಷೆ_ಕೀಲಿ, std::shared_ptr<ov::InferRequest>, ನಕ್ಷೆ_ಕೀಲಿ_hash> infer_request_cache_prefill;
    std::unordered_map<ನಕ್ಷೆ_ಕೀಲಿ, std::ಸದಿಶ<std::string>, ನಕ್ಷೆ_ಕೀಲಿ_hash> ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_cache;
    std::unordered_map<ನಕ್ಷೆ_ಕೀಲಿ, std::ಸದಿಶ<std::string>, ನಕ್ಷೆ_ಕೀಲಿ_hash> ov_output_ಹೆಸರುಗಳು_cache;
    size_t stateful_kv_ಗಾತ್ರ;
    std::map<std::string, std::string> kv_state_ಒಳಹರಿವು_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    ರಚನೆ compute_ನಕ್ಷೆ {
        ggml_cgraph ನಕ್ಷೆ;
        std::ಸದಿಶ<ggml_tensor *> nodes;
    };
    std::unordered_map<ಸ್ಥಿರಾಂಕ ggml_cgraph *, compute_ನಕ್ಷೆ> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_ಗಾತ್ರ(0) {}

    ಶೂನ್ಯ clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_ಒಳಹರಿವು_ಹೆಸರುಗಳು_cache.clear();
        ov_output_ಹೆಸರುಗಳು_cache.clear();
        kv_state_ಒಳಹರಿವು_name_map.clear();
        stateful_kv_ಗಾತ್ರ = 0;
    }

    ಶೂನ್ಯ clear_caches() {
        std::ಬೀಗ_guard<std::mutex> ಬೀಗ(ctx_mutex);
        clear_caches_locked();
    }
};

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ov_ನಕ್ಷೆ_compute(ರಚನೆ ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2);

bool save_ggml_tensor_ದತ್ತಾಂಶ_ಗಮ್ಯ_txt(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ std::string & file_ಮಾರ್ಗ);

ಶೂನ್ಯ print_ಒಳಹರಿವು_tensor_info(ಸ್ಥಿರಾಂಕ std::string & name, ಸ್ಥಿರಾಂಕ ov::Tensor & tensor);

ಶೂನ್ಯ print_output_tensor_info(ಸ್ಥಿರಾಂಕ std::string & name, ಸ್ಥಿರಾಂಕ ov::Tensor & tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * output_dst);

ಮಾದರಿ <typename T>
std::ಸದಿಶ<T> pad_ಒಳಹರಿವು(ಸ್ಥಿರಾಂಕ T * ದತ್ತಾಂಶ,
                         size_t ಅಡ್ಡಸಾಲುಗಳು,
                         size_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2,
                         size_t padded_ಅಡ್ಡಸಾಲುಗಳು,
                         size_t padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
                         T pad_ಮೌಲ್ಯ) {
    std::ಸದಿಶ<T> padded(padded_ಅಡ್ಡಸಾಲುಗಳು * padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, pad_ಮೌಲ್ಯ);

    ಗಾಗಿ (size_t i = 0; i < std::min(ಅಡ್ಡಸಾಲುಗಳು, padded_ಅಡ್ಡಸಾಲುಗಳು); ++i) {
        ಗಾಗಿ (size_t j = 0; j < std::min(ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ); ++j) {
            padded[i * padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ + j] = ದತ್ತಾಂಶ[i * ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 + j];
        }
    }

    ಹಿಂತಿರುಗಿ padded;
}

ಮಾದರಿ <typename T>
std::ಸದಿಶ<T> pad_ಒಳಹರಿವು(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, size_t padded_ಅಡ್ಡಸಾಲುಗಳು, size_t padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, T pad_ಮೌಲ್ಯ) {
    ಹಿಂತಿರುಗಿ pad_ಒಳಹರಿವು<T>(reinterpret_cast<ಸ್ಥಿರಾಂಕ T *>(tensor->ದತ್ತಾಂಶ),
                        static_cast<size_t>(tensor->ne[1]),  //ಅಡ್ಡಸಾಲುಗಳು
                        static_cast<size_t>(tensor->ne[0]),  //ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ
                        padded_ಅಡ್ಡಸಾಲುಗಳು, padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, pad_ಮೌಲ್ಯ);
}

ಸ್ಥಿರಾಂಕ ggml_tensor * get_inp_pos_tensor(ರಚನೆ ggml_cgraph * cgraph);

int64_t get_inp_pos_n_ಟೋಕನುಗಳು(ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ggml_tensor * inp_pos);

bool get_is_prefill(ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ggml_tensor * inp_pos);

bool is_naive(ರಚನೆ ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(ರಚನೆ ggml_cgraph * cgraph);

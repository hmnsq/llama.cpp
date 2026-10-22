#समावेश "ggml-decoder.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-openvino-extra.h"

#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <functional>
#समावेश <memory>
#समावेश <mutex>
#समावेश <openvino/runtime/core.hpp>
#समावेश <openvino/runtime/infer_request.hpp>
#समावेश <string>
#समावेश <unordered_map>
#समावेश <unordered_set>
#समावेश <utility>
#समावेश <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
काष्ठा graph_कुंजी {
    पूर्णांक n_nodes;
    पूर्णांक n_leaves;
    std::string first_गाँठ_name;
    std::string last_गाँठ_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> निवेश_srcs;

    graph_कुंजी(स्थिर ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        अगर (n_nodes > 0) {
            first_गाँठ_name = cgraph->nodes[0]->name;
            last_गाँठ_name = cgraph->nodes[n_nodes - 1]->name;
        }

        अटल स्थिर bool full_कुंजी = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        अगर (!full_कुंजी && !include_inputs) {
            वापस;
        }

        std::unordered_set<स्थिर ggml_tensor *> गाँठ_set;
        गाँठ_set.reserve(cgraph->n_nodes);
        क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
            गाँठ_set.जोड़ो(cgraph->nodes[i]);
        }

        क्रम (पूर्णांक गाँठ_idx = 0; गाँठ_idx < cgraph->n_nodes; गाँठ_idx++) {
            स्थिर ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idx];
            क्रम (पूर्णांक src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                स्थिर ggml_tensor * src = गाँठ->src[src_idx];
                अगर (src == nullptr || src->बफर == nullptr || गाँठ_set.गिनती(src) || src->बफर->usage == GGML_BACKEND_बफर_USAGE_WEIGHTS) {
                    जारी;
                }

                निवेश_srcs.धकेलो_back(std::to_string(गाँठ_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(गाँठ));
            }
        }
    }

    bool चालक==(स्थिर graph_कुंजी & other) स्थिर {
        वापस n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_गाँठ_name == other.first_गाँठ_name &&
               last_गाँठ_name == other.last_गाँठ_name && निवेश_srcs == other.निवेश_srcs;
    }
};

काष्ठा graph_कुंजी_hash {
    माप_प्रकार चालक()(स्थिर graph_कुंजी & कुंजी) स्थिर {
        माप_प्रकार hash = std::hash<पूर्णांक>{}(कुंजी.n_nodes);
        hash ^= std::hash<पूर्णांक>{}(कुंजी.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        अगर (कुंजी.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(कुंजी.first_गाँठ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(कुंजी.last_गाँठ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        क्रम (स्थिर स्वतः & s : कुंजी.निवेश_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        वापस hash;
    }
};

काष्ठा decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

काष्ठा ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> निवेश_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
काष्ठा ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    माप_प्रकार backend_गिनती = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
काष्ठा ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_कुंजी, std::shared_ptr<decoder_runtime_ctx>, graph_कुंजी_hash> decoder_cache;
    std::unordered_map<graph_कुंजी, std::shared_ptr<ov::InferRequest>, graph_कुंजी_hash> infer_request_cache;
    std::unordered_map<graph_कुंजी, std::shared_ptr<ov::InferRequest>, graph_कुंजी_hash> infer_request_cache_prefill;
    std::unordered_map<graph_कुंजी, std::vector<std::string>, graph_कुंजी_hash> ov_निवेश_names_cache;
    std::unordered_map<graph_कुंजी, std::vector<std::string>, graph_कुंजी_hash> ov_output_names_cache;
    माप_प्रकार stateful_kv_आकार;
    std::map<std::string, std::string> kv_state_निवेश_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    काष्ठा compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<स्थिर ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_आकार(0) {}

    व्योम clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_निवेश_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_निवेश_name_map.clear();
        stateful_kv_आकार = 0;
    }

    व्योम clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

क्रमागत ggml_status ov_graph_compute(काष्ठा ggml_cgraph * cgraph, ggml_backend_t backend);

माप_प्रकार checksum(स्थिर व्योम * तथ्य, माप_प्रकार आकार);

bool save_ggml_tensor_तथ्य_to_txt(स्थिर ggml_tensor * tensor, स्थिर std::string & file_path);

व्योम print_निवेश_tensor_info(स्थिर std::string & name, स्थिर ov::Tensor & tensor);

व्योम print_output_tensor_info(स्थिर std::string & name, स्थिर ov::Tensor & tensor, स्थिर व्योम * output_dst);

ढाँचा <typename T>
std::vector<T> pad_निवेश(स्थिर T * तथ्य,
                         माप_प्रकार rows,
                         माप_प्रकार cols,
                         माप_प्रकार padded_rows,
                         माप_प्रकार padded_cols,
                         T pad_मूल्य) {
    std::vector<T> padded(padded_rows * padded_cols, pad_मूल्य);

    क्रम (माप_प्रकार i = 0; i < std::min(rows, padded_rows); ++i) {
        क्रम (माप_प्रकार j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = तथ्य[i * cols + j];
        }
    }

    वापस padded;
}

ढाँचा <typename T>
std::vector<T> pad_निवेश(स्थिर ggml_tensor * tensor, माप_प्रकार padded_rows, माप_प्रकार padded_cols, T pad_मूल्य) {
    वापस pad_निवेश<T>(reinterpret_cast<स्थिर T *>(tensor->तथ्य),
                        static_cast<माप_प्रकार>(tensor->ne[1]),  // rows
                        static_cast<माप_प्रकार>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_मूल्य);
}

स्थिर ggml_tensor * get_inp_pos_tensor(काष्ठा ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(काष्ठा ggml_cgraph * cgraph, स्थिर ggml_tensor * inp_pos);

bool get_is_prefill(काष्ठा ggml_cgraph * cgraph, स्थिर ggml_tensor * inp_pos);

bool is_naive(काष्ठा ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(काष्ठा ggml_cgraph * cgraph);

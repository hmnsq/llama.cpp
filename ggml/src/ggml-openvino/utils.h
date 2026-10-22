#अन्तर्भाव "ggml-decoder.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-openvino-extra.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <functional>
#अन्तर्भाव <memory>
#अन्तर्भाव <mutex>
#अन्तर्भाव <openvino/runtime/core.hpp>
#अन्तर्भाव <openvino/runtime/infer_request.hpp>
#अन्तर्भाव <string>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <utility>
#अन्तर्भाव <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
संरचना graph_साँचो {
    पूर्णाङ्क n_nodes;
    पूर्णाङ्क n_leaves;
    std::string first_गाँठो_name;
    std::string last_गाँठो_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> आगत_srcs;

    graph_साँचो(अचर ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        यदि (n_nodes > 0) {
            first_गाँठो_name = cgraph->nodes[0]->name;
            last_गाँठो_name = cgraph->nodes[n_nodes - 1]->name;
        }

        स्थिर अचर bool full_साँचो = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        यदि (!full_साँचो && !include_inputs) {
            फर्काऊ;
        }

        std::unordered_set<अचर ggml_tensor *> गाँठो_set;
        गाँठो_set.reserve(cgraph->n_nodes);
        लागि (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
            गाँठो_set.घुसाउ(cgraph->nodes[i]);
        }

        लागि (पूर्णाङ्क गाँठो_idx = 0; गाँठो_idx < cgraph->n_nodes; गाँठो_idx++) {
            अचर ggml_tensor * गाँठो = cgraph->nodes[गाँठो_idx];
            लागि (पूर्णाङ्क src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                अचर ggml_tensor * src = गाँठो->src[src_idx];
                यदि (src == nullptr || src->buffer == nullptr || गाँठो_set.गन्ती(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    जारी;
                }

                आगत_srcs.थप_back(std::to_string(गाँठो_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(गाँठो));
            }
        }
    }

    bool चालक==(अचर graph_साँचो & other) अचर {
        फर्काऊ n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_गाँठो_name == other.first_गाँठो_name &&
               last_गाँठो_name == other.last_गाँठो_name && आगत_srcs == other.आगत_srcs;
    }
};

संरचना graph_साँचो_hash {
    माप_प्रकार चालक()(अचर graph_साँचो & साँचो) अचर {
        माप_प्रकार hash = std::hash<पूर्णाङ्क>{}(साँचो.n_nodes);
        hash ^= std::hash<पूर्णाङ्क>{}(साँचो.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        यदि (साँचो.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(साँचो.first_गाँठो_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(साँचो.last_गाँठो_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        लागि (अचर स्वतः & s : साँचो.आगत_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        फर्काऊ hash;
    }
};

संरचना decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

संरचना ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> आगत_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
संरचना ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    माप_प्रकार backend_गन्ती = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
संरचना ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_साँचो, std::shared_ptr<decoder_runtime_ctx>, graph_साँचो_hash> decoder_cache;
    std::unordered_map<graph_साँचो, std::shared_ptr<ov::InferRequest>, graph_साँचो_hash> infer_request_cache;
    std::unordered_map<graph_साँचो, std::shared_ptr<ov::InferRequest>, graph_साँचो_hash> infer_request_cache_prefill;
    std::unordered_map<graph_साँचो, std::vector<std::string>, graph_साँचो_hash> ov_आगत_names_cache;
    std::unordered_map<graph_साँचो, std::vector<std::string>, graph_साँचो_hash> ov_output_names_cache;
    माप_प्रकार stateful_kv_size;
    std::map<std::string, std::string> kv_state_आगत_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    संरचना compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<अचर ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    शून्य clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_आगत_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_आगत_name_map.clear();
        stateful_kv_size = 0;
    }

    शून्य clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

गणना ggml_status ov_graph_compute(संरचना ggml_cgraph * cgraph, ggml_backend_t backend);

माप_प्रकार checksum(अचर शून्य * data, माप_प्रकार size);

bool save_ggml_tensor_data_to_txt(अचर ggml_tensor * tensor, अचर std::string & file_path);

शून्य print_आगत_tensor_info(अचर std::string & name, अचर ov::Tensor & tensor);

शून्य print_output_tensor_info(अचर std::string & name, अचर ov::Tensor & tensor, अचर शून्य * output_dst);

ढाँचा <typename T>
std::vector<T> pad_आगत(अचर T * data,
                         माप_प्रकार rows,
                         माप_प्रकार cols,
                         माप_प्रकार padded_rows,
                         माप_प्रकार padded_cols,
                         T pad_मान) {
    std::vector<T> padded(padded_rows * padded_cols, pad_मान);

    लागि (माप_प्रकार i = 0; i < std::min(rows, padded_rows); ++i) {
        लागि (माप_प्रकार j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    फर्काऊ padded;
}

ढाँचा <typename T>
std::vector<T> pad_आगत(अचर ggml_tensor * tensor, माप_प्रकार padded_rows, माप_प्रकार padded_cols, T pad_मान) {
    फर्काऊ pad_आगत<T>(reinterpret_cast<अचर T *>(tensor->data),
                        static_cast<माप_प्रकार>(tensor->ne[1]),  // rows
                        static_cast<माप_प्रकार>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_मान);
}

अचर ggml_tensor * get_inp_pos_tensor(संरचना ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(संरचना ggml_cgraph * cgraph, अचर ggml_tensor * inp_pos);

bool get_is_prefill(संरचना ggml_cgraph * cgraph, अचर ggml_tensor * inp_pos);

bool is_naive(संरचना ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(संरचना ggml_cgraph * cgraph);

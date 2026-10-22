#ଅନ୍ତର୍ଭାଵ "ggml-decoder.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-openvino-extra.h"

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <functional>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <mutex>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/core.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/runtime/infer_request.hpp>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <unordered_set>
#ଅନ୍ତର୍ଭାଵ <utility>
#ଅନ୍ତର୍ଭାଵ <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
ସଂରଚନା graph_ଚାବି {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_nodes;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_leaves;
    std::string first_ଗଣ୍ଠି_name;
    std::string last_ଗଣ୍ଠି_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> ନିବେଶ_srcs;

    graph_ଚାବି(ଅଚଳ ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        ଯଦି (n_nodes > 0) {
            first_ଗଣ୍ଠି_name = cgraph->nodes[0]->name;
            last_ଗଣ୍ଠି_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ସ୍ଥିର ଅଚଳ bool full_ଚାବି = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        ଯଦି (!full_ଚାବି && !include_inputs) {
            ଫେରନ୍ତୁ;
        }

        std::unordered_set<ଅଚଳ ggml_tensor *> ଗଣ୍ଠି_set;
        ଗଣ୍ଠି_set.reserve(cgraph->n_nodes);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
            ଗଣ୍ଠି_set.ଭର୍ତ୍ତି(cgraph->nodes[i]);
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx = 0; ଗଣ୍ଠି_idx < cgraph->n_nodes; ଗଣ୍ଠି_idx++) {
            ଅଚଳ ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_idx];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ଅଚଳ ggml_tensor * src = ଗଣ୍ଠି->src[src_idx];
                ଯଦି (src == nullptr || src->buffer == nullptr || ଗଣ୍ଠି_set.ଗଣନା_2(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    ଜାରି;
                }

                ନିବେଶ_srcs.ଠେଲ_back(std::to_string(ଗଣ୍ଠି_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(ଗଣ୍ଠି));
            }
        }
    }

    bool operator==(ଅଚଳ graph_ଚାବି & other) ଅଚଳ {
        ଫେରନ୍ତୁ n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_ଗଣ୍ଠି_name == other.first_ଗଣ୍ଠି_name &&
               last_ଗଣ୍ଠି_name == other.last_ଗଣ୍ଠି_name && ନିବେଶ_srcs == other.ନିବେଶ_srcs;
    }
};

ସଂରଚନା graph_ଚାବି_hash {
    size_t operator()(ଅଚଳ graph_ଚାବି & ଚାବି) ଅଚଳ {
        size_t hash = std::hash<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>{}(ଚାବି.n_nodes);
        hash ^= std::hash<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>{}(ଚାବି.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        ଯଦି (ଚାବି.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(ଚାବି.first_ଗଣ୍ଠି_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(ଚାବି.last_ଗଣ୍ଠି_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        ପାଇଁ (ଅଚଳ auto & s : ଚାବି.ନିବେଶ_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        ଫେରନ୍ତୁ hash;
    }
};

ସଂରଚନା decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

ସଂରଚନା ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> ନିବେଶ_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
ସଂରଚନା ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_ଗଣନା = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
ସଂରଚନା ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_ଚାବି, std::shared_ptr<decoder_runtime_ctx>, graph_ଚାବି_hash> decoder_cache;
    std::unordered_map<graph_ଚାବି, std::shared_ptr<ov::InferRequest>, graph_ଚାବି_hash> infer_request_cache;
    std::unordered_map<graph_ଚାବି, std::shared_ptr<ov::InferRequest>, graph_ଚାବି_hash> infer_request_cache_prefill;
    std::unordered_map<graph_ଚାବି, std::vector<std::string>, graph_ଚାବି_hash> ov_ନିବେଶ_names_cache;
    std::unordered_map<graph_ଚାବି, std::vector<std::string>, graph_ଚାବି_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_ନିବେଶ_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    ସଂରଚନା compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<ଅଚଳ ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    ଶୂନ୍ୟ clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_ନିବେଶ_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_ନିବେଶ_name_map.clear();
        stateful_kv_size = 0;
    }

    ଶୂନ୍ୟ clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

ଗଣନା ggml_status ov_graph_compute(ସଂରଚନା ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ଅଚଳ ଶୂନ୍ୟ * data, size_t size);

bool save_ggml_tensor_data_to_txt(ଅଚଳ ggml_tensor * tensor, ଅଚଳ std::string & file_path);

ଶୂନ୍ୟ print_ନିବେଶ_tensor_info(ଅଚଳ std::string & name, ଅଚଳ ov::Tensor & tensor);

ଶୂନ୍ୟ print_output_tensor_info(ଅଚଳ std::string & name, ଅଚଳ ov::Tensor & tensor, ଅଚଳ ଶୂନ୍ୟ * output_dst);

template <typename T>
std::vector<T> pad_ନିବେଶ(ଅଚଳ T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_ମୂଲ୍ୟ) {
    std::vector<T> padded(padded_rows * padded_cols, pad_ମୂଲ୍ୟ);

    ପାଇଁ (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        ପାଇଁ (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    ଫେରନ୍ତୁ padded;
}

template <typename T>
std::vector<T> pad_ନିବେଶ(ଅଚଳ ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_ମୂଲ୍ୟ) {
    ଫେରନ୍ତୁ pad_ନିବେଶ<T>(reinterpret_cast<ଅଚଳ T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_ମୂଲ୍ୟ);
}

ଅଚଳ ggml_tensor * get_inp_pos_tensor(ସଂରଚନା ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ggml_tensor * inp_pos);

bool get_is_prefill(ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ggml_tensor * inp_pos);

bool is_naive(ସଂରଚନା ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(ସଂରଚନା ggml_cgraph * cgraph);

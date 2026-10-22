#অন্তর্ভাব "ggml-decoder.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-openvino-extra.h"

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <functional>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <openvino/runtime/core.hpp>
#অন্তর্ভাব <openvino/runtime/infer_request.hpp>
#অন্তর্ভাব <string>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <utility>
#অন্তর্ভাব <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
সংরচনা graph_চাবি {
    পূর্ণাঙ্ক n_nodes;
    পূর্ণাঙ্ক n_leaves;
    std::string first_গ্রন্থি_name;
    std::string last_গ্রন্থি_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> প্রবেশ_srcs;

    graph_চাবি(ধ্রুব ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        যদি (n_nodes > 0) {
            first_গ্রন্থি_name = cgraph->nodes[0]->name;
            last_গ্রন্থি_name = cgraph->nodes[n_nodes - 1]->name;
        }

        স্থির ধ্রুব bool full_চাবি = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        যদি (!full_চাবি && !include_inputs) {
            প্রত্যাগম;
        }

        std::unordered_set<ধ্রুব ggml_tensor *> গ্রন্থি_set;
        গ্রন্থি_set.reserve(cgraph->n_nodes);
        প্রতি (পূর্ণাঙ্ক i = 0; i < cgraph->n_nodes; i++) {
            গ্রন্থি_set.ঢোকাও(cgraph->nodes[i]);
        }

        প্রতি (পূর্ণাঙ্ক গ্রন্থি_idx = 0; গ্রন্থি_idx < cgraph->n_nodes; গ্রন্থি_idx++) {
            ধ্রুব ggml_tensor * গ্রন্থি = cgraph->nodes[গ্রন্থি_idx];
            প্রতি (পূর্ণাঙ্ক src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ধ্রুব ggml_tensor * src = গ্রন্থি->src[src_idx];
                যদি (src == nullptr || src->buffer == nullptr || গ্রন্থি_set.গুনতি(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    অনুবর্ত;
                }

                প্রবেশ_srcs.ঠেলো_back(std::to_string(গ্রন্থি_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(গ্রন্থি));
            }
        }
    }

    bool operator==(ধ্রুব graph_চাবি & other) ধ্রুব {
        প্রত্যাগম n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_গ্রন্থি_name == other.first_গ্রন্থি_name &&
               last_গ্রন্থি_name == other.last_গ্রন্থি_name && প্রবেশ_srcs == other.প্রবেশ_srcs;
    }
};

সংরচনা graph_চাবি_hash {
    size_t operator()(ধ্রুব graph_চাবি & চাবি) ধ্রুব {
        size_t hash = std::hash<পূর্ণাঙ্ক>{}(চাবি.n_nodes);
        hash ^= std::hash<পূর্ণাঙ্ক>{}(চাবি.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        যদি (চাবি.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(চাবি.first_গ্রন্থি_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(চাবি.last_গ্রন্থি_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        প্রতি (ধ্রুব auto & s : চাবি.প্রবেশ_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        প্রত্যাগম hash;
    }
};

সংরচনা decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

সংরচনা ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> প্রবেশ_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
সংরচনা ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_গুনতি = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
সংরচনা ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_চাবি, std::shared_ptr<decoder_runtime_ctx>, graph_চাবি_hash> decoder_cache;
    std::unordered_map<graph_চাবি, std::shared_ptr<ov::InferRequest>, graph_চাবি_hash> infer_request_cache;
    std::unordered_map<graph_চাবি, std::shared_ptr<ov::InferRequest>, graph_চাবি_hash> infer_request_cache_prefill;
    std::unordered_map<graph_চাবি, std::vector<std::string>, graph_চাবি_hash> ov_প্রবেশ_names_cache;
    std::unordered_map<graph_চাবি, std::vector<std::string>, graph_চাবি_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_প্রবেশ_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    সংরচনা compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<ধ্রুব ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    শূন্য clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_প্রবেশ_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_প্রবেশ_name_map.clear();
        stateful_kv_size = 0;
    }

    শূন্য clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

গণনা ggml_status ov_graph_compute(সংরচনা ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ধ্রুব শূন্য * data, size_t size);

bool save_ggml_tensor_data_to_txt(ধ্রুব ggml_tensor * tensor, ধ্রুব std::string & file_path);

শূন্য print_প্রবেশ_tensor_info(ধ্রুব std::string & name, ধ্রুব ov::Tensor & tensor);

শূন্য print_output_tensor_info(ধ্রুব std::string & name, ধ্রুব ov::Tensor & tensor, ধ্রুব শূন্য * output_dst);

template <typename T>
std::vector<T> pad_প্রবেশ(ধ্রুব T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_মান) {
    std::vector<T> padded(padded_rows * padded_cols, pad_মান);

    প্রতি (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        প্রতি (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    প্রত্যাগম padded;
}

template <typename T>
std::vector<T> pad_প্রবেশ(ধ্রুব ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_মান) {
    প্রত্যাগম pad_প্রবেশ<T>(reinterpret_cast<ধ্রুব T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_মান);
}

ধ্রুব ggml_tensor * get_inp_pos_tensor(সংরচনা ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(সংরচনা ggml_cgraph * cgraph, ধ্রুব ggml_tensor * inp_pos);

bool get_is_prefill(সংরচনা ggml_cgraph * cgraph, ধ্রুব ggml_tensor * inp_pos);

bool is_naive(সংরচনা ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(সংরচনা ggml_cgraph * cgraph);

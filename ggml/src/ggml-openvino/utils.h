#شامل "ggml-decoder.h"
#شامل "ggml-impl.h"
#شامل "ggml-openvino-extra.h"

#شامل <cstddef>
#شامل <cstdint>
#شامل <functional>
#شامل <memory>
#شامل <mutex>
#شامل <openvino/runtime/core.hpp>
#شامل <openvino/runtime/infer_request.hpp>
#شامل <string>
#شامل <unordered_map>
#شامل <unordered_set>
#شامل <utility>
#شامل <vector>

// Cache key for a translated/compiled graph. Node count plus the two end node names identify a
// graph during inference, where the same few graphs repeat for the whole session. The list of
// external input names below tells those apart more precisely, but it walks every node and every
// src slot, so it is built only when GGML_OPENVINO_FULL_GRAPH_KEY is set. Op tests run many small
// graphs that can share a node count and end names, and need the full key.
ساختار graph_کلید {
    صحیح n_nodes;
    صحیح n_leaves;
    std::string first_گره_name;
    std::string last_گره_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::vector<std::string> درون_داد_srcs;

    graph_کلید(ثابت ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        اگر (n_nodes > 0) {
            first_گره_name = cgraph->nodes[0]->name;
            last_گره_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ایستا ثابت bool full_کلید = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        اگر (!full_کلید && !include_inputs) {
            بازگشت;
        }

        std::unordered_set<ثابت ggml_tensor *> گره_set;
        گره_set.reserve(cgraph->n_nodes);
        برای (صحیح i = 0; i < cgraph->n_nodes; i++) {
            گره_set.درج(cgraph->nodes[i]);
        }

        برای (صحیح گره_idx = 0; گره_idx < cgraph->n_nodes; گره_idx++) {
            ثابت ggml_tensor * گره = cgraph->nodes[گره_idx];
            برای (صحیح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ثابت ggml_tensor * src = گره->src[src_idx];
                اگر (src == nullptr || src->buffer == nullptr || گره_set.شمار(src) || src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    ادامه;
                }

                درون_داد_srcs.افزودن_back(std::to_string(گره_idx) + ":" + std::to_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(گره));
            }
        }
    }

    bool operator==(ثابت graph_کلید & other) ثابت {
        بازگشت n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               first_گره_name == other.first_گره_name &&
               last_گره_name == other.last_گره_name && درون_داد_srcs == other.درون_داد_srcs;
    }
};

ساختار graph_کلید_hash {
    size_t operator()(ثابت graph_کلید & کلید) ثابت {
        size_t hash = std::hash<صحیح>{}(کلید.n_nodes);
        hash ^= std::hash<صحیح>{}(کلید.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        اگر (کلید.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(کلید.first_گره_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(کلید.last_گره_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        برای (ثابت auto & s : کلید.درون_داد_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        بازگشت hash;
    }
};

ساختار decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

ساختار ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> درون_داد_names;
    std::vector<std::string> output_names;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
ساختار ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_شمار = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
ساختار ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_کلید, std::shared_ptr<decoder_runtime_ctx>, graph_کلید_hash> decoder_cache;
    std::unordered_map<graph_کلید, std::shared_ptr<ov::InferRequest>, graph_کلید_hash> infer_request_cache;
    std::unordered_map<graph_کلید, std::shared_ptr<ov::InferRequest>, graph_کلید_hash> infer_request_cache_prefill;
    std::unordered_map<graph_کلید, std::vector<std::string>, graph_کلید_hash> ov_درون_داد_names_cache;
    std::unordered_map<graph_کلید, std::vector<std::string>, graph_کلید_hash> ov_output_names_cache;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_درون_داد_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    ساختار compute_graph {
        ggml_cgraph graph;
        std::vector<ggml_tensor *> nodes;
    };
    std::unordered_map<ثابت ggml_cgraph *, compute_graph> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    تهی clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_درون_داد_names_cache.clear();
        ov_output_names_cache.clear();
        kv_state_درون_داد_name_map.clear();
        stateful_kv_size = 0;
    }

    تهی clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

شمارش ggml_status ov_graph_compute(ساختار ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ثابت تهی * data, size_t size);

bool save_ggml_tensor_data_to_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_path);

تهی print_درون_داد_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor);

تهی print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت تهی * output_dst);

template <typename T>
std::vector<T> pad_درون_داد(ثابت T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_مقدار) {
    std::vector<T> padded(padded_rows * padded_cols, pad_مقدار);

    برای (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        برای (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    بازگشت padded;
}

template <typename T>
std::vector<T> pad_درون_داد(ثابت ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_مقدار) {
    بازگشت pad_درون_داد<T>(reinterpret_cast<ثابت T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_مقدار);
}

ثابت ggml_tensor * get_inp_pos_tensor(ساختار ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(ساختار ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool get_is_prefill(ساختار ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool is_naive(ساختار ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(ساختار ggml_cgraph * cgraph);

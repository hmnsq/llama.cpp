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
ساخت گراف_کلید {
    صحیح n_nodes;
    صحیح n_leaves;
    std::string پہلا_گرہ_name;
    std::string آخری_گرہ_name;
    // Each entry: "<node_idx>:<src_idx>:<op_type>"
    std::سمتیہ<std::string> داخلہ_srcs;

    گراف_کلید(ثابت ggml_cgraph * cgraph, bool include_inputs = false) : n_nodes(cgraph->n_nodes), n_leaves(cgraph->n_leafs) {
        اگر (n_nodes > 0) {
            پہلا_گرہ_name = cgraph->nodes[0]->name;
            آخری_گرہ_name = cgraph->nodes[n_nodes - 1]->name;
        }

        ساکن ثابت bool بھرا_کلید = ggml_openvino_getenv_int("GGML_OPENVINO_FULL_GRAPH_KEY") != 0;
        اگر (!بھرا_کلید && !include_inputs) {
            واپس;
        }

        std::unordered_سیٹ<ثابت ggml_tensor *> گرہ_سیٹ;
        گرہ_سیٹ.reserve(cgraph->n_nodes);
        برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
            گرہ_سیٹ.داخل_کرو(cgraph->nodes[i]);
        }

        برائے (صحیح گرہ_idx = 0; گرہ_idx < cgraph->n_nodes; گرہ_idx++) {
            ثابت ggml_tensor * گرہ = cgraph->nodes[گرہ_idx];
            برائے (صحیح src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ثابت ggml_tensor * src = گرہ->src[src_idx];
                اگر (src == nullptr || src->بفر == nullptr || گرہ_سیٹ.گنتی(src) || src->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS) {
                    جاری;
                }

                داخلہ_srcs.ڈالو_عقب(std::مقصد_string(گرہ_idx) + ":" + std::مقصد_string(src_idx) + ":" +
                    GgmlOvDecoder::compute_op_type(گرہ));
            }
        }
    }

    bool عامل==(ثابت گراف_کلید & other) ثابت {
        واپس n_nodes == other.n_nodes && n_leaves == other.n_leaves &&
               پہلا_گرہ_name == other.پہلا_گرہ_name &&
               آخری_گرہ_name == other.آخری_گرہ_name && داخلہ_srcs == other.داخلہ_srcs;
    }
};

ساخت گراف_کلید_hash {
    size_t عامل()(ثابت گراف_کلید & کلید) ثابت {
        size_t hash = std::hash<صحیح>{}(کلید.n_nodes);
        hash ^= std::hash<صحیح>{}(کلید.n_leaves) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        اگر (کلید.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(کلید.پہلا_گرہ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(کلید.آخری_گرہ_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        برائے (ثابت خودکار & s : کلید.داخلہ_srcs) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        واپس hash;
    }
};

ساخت decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

ساخت ov_compiled_گراف {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::سمتیہ<std::string> داخلہ_نام;
    std::سمتیہ<std::string> output_نام;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
ساخت ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_گراف> graphs;
    size_t backend_گنتی = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
ساخت ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<گراف_کلید, std::shared_ptr<decoder_runtime_ctx>, گراف_کلید_hash> decoder_cache;
    std::unordered_map<گراف_کلید, std::shared_ptr<ov::InferRequest>, گراف_کلید_hash> infer_request_cache;
    std::unordered_map<گراف_کلید, std::shared_ptr<ov::InferRequest>, گراف_کلید_hash> infer_request_cache_prefill;
    std::unordered_map<گراف_کلید, std::سمتیہ<std::string>, گراف_کلید_hash> ov_داخلہ_نام_cache;
    std::unordered_map<گراف_کلید, std::سمتیہ<std::string>, گراف_کلید_hash> ov_output_نام_cache;
    size_t stateful_kv_سائز;
    std::map<std::string, std::string> kv_state_داخلہ_name_map;
    // compute-only copies of graphs that carry unselected ggml_build_forward_select() branches
    ساخت compute_گراف {
        ggml_cgraph گراف;
        std::سمتیہ<ggml_tensor *> nodes;
    };
    std::unordered_map<ثابت ggml_cgraph *, compute_گراف> compute_graphs;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_سائز(0) {}

    خالی clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_داخلہ_نام_cache.clear();
        ov_output_نام_cache.clear();
        kv_state_داخلہ_name_map.clear();
        stateful_kv_سائز = 0;
    }

    خالی clear_caches() {
        std::تالا_guard<std::mutex> تالا(ctx_mutex);
        clear_caches_locked();
    }
};

شمار ggml_حالت ov_گراف_compute(ساخت ggml_cgraph * cgraph, ggml_backend_t backend);

size_t checksum(ثابت خالی * ڈیٹا, size_t سائز);

bool save_ggml_tensor_ڈیٹا_مقصد_txt(ثابت ggml_tensor * tensor, ثابت std::string & file_راستہ);

خالی print_داخلہ_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor);

خالی print_output_tensor_info(ثابت std::string & name, ثابت ov::Tensor & tensor, ثابت خالی * output_dst);

سانچہ <typename T>
std::سمتیہ<T> pad_داخلہ(ثابت T * ڈیٹا,
                         size_t صفیں,
                         size_t ستون_تعداد_2,
                         size_t padded_صفیں,
                         size_t padded_ستون_تعداد,
                         T pad_قدر) {
    std::سمتیہ<T> padded(padded_صفیں * padded_ستون_تعداد, pad_قدر);

    برائے (size_t i = 0; i < std::min(صفیں, padded_صفیں); ++i) {
        برائے (size_t j = 0; j < std::min(ستون_تعداد_2, padded_ستون_تعداد); ++j) {
            padded[i * padded_ستون_تعداد + j] = ڈیٹا[i * ستون_تعداد_2 + j];
        }
    }

    واپس padded;
}

سانچہ <typename T>
std::سمتیہ<T> pad_داخلہ(ثابت ggml_tensor * tensor, size_t padded_صفیں, size_t padded_ستون_تعداد, T pad_قدر) {
    واپس pad_داخلہ<T>(reinterpret_cast<ثابت T *>(tensor->ڈیٹا),
                        static_cast<size_t>(tensor->ne[1]),  //صفیں
                        static_cast<size_t>(tensor->ne[0]),  //ستون_تعداد
                        padded_صفیں, padded_ستون_تعداد, pad_قدر);
}

ثابت ggml_tensor * get_inp_pos_tensor(ساخت ggml_cgraph * cgraph);

int64_t get_inp_pos_n_علامات(ساخت ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool get_is_prefill(ساخت ggml_cgraph * cgraph, ثابت ggml_tensor * inp_pos);

bool is_naive(ساخت ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(ساخت ggml_cgraph * cgraph);

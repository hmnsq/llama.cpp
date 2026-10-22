#অন্তর্ভাব "backend/shared/apir_cs_rpc.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-remoting.h"

#অন্তর্ভাব <cinttypes>
#অন্তর্ভাব <unordered_map>
#অন্তর্ভাব <unordered_set>
#অন্তর্ভাব <vector>

apir_rpc_tensor apir_serialize_tensor(ধ্রুব ggml_tensor * tensor) {
    apir_rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    যদি (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;

        result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
    } অন্যথা {
        result.buffer = 0;
    }
    প্রতি (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    প্রতি (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / পরিমাণ(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    প্রতি (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    যদি (tensor->data) {
        যদি (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    প্রত্যাগম result;
}

শূন্য apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    যদি (tensor == nullptr) {
        প্রত্যাগম;
    }
    যদি (visited.find(tensor) != visited.end()) {
        প্রত্যাগম;
    }
    visited.insert(tensor);
    প্রতি (পূর্ণাঙ্ক i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

শূন্য apir_serialize_graph(ধ্রুব ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    প্রতি (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    পূর্ণাঙ্ক      output_size =
        পরিমাণ(uint32_t) + n_nodes * পরিমাণ(uint64_t) + পরিমাণ(uint32_t) + n_tensors * পরিমাণ(apir_rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, পরিমাণ(n_nodes));
    প্রতি (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + পরিমাণ(n_nodes) + i * পরিমাণ(uint64_t), &cgraph->nodes[i], পরিমাণ(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + পরিমাণ(n_nodes) + n_nodes * পরিমাণ(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.data() + পরিমাণ(n_nodes) + n_nodes * পরিমাণ(uint64_t) + পরিমাণ(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * পরিমাণ(apir_rpc_tensor));
}

#അന്തര്ഭാവ "backend/shared/apir_cs_rpc.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-remoting.h"

#അന്തര്ഭാവ <cinttypes>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <unordered_set>
#അന്തര്ഭാവ <vector>

apir_rpc_tensor apir_serialize_tensor(സ്ഥിരം ggml_tensor * tensor) {
    apir_rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    എങ്കിൽ (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;

        result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
    } അല്ലെങ്കിൽ {
        result.buffer = 0;
    }
    വേണ്ടി (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    വേണ്ടി (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / വലുപ്പം(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    വേണ്ടി (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    എങ്കിൽ (tensor->data) {
        എങ്കിൽ (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    തിരികെ result;
}

ശൂന്യം apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    എങ്കിൽ (tensor == nullptr) {
        തിരികെ;
    }
    എങ്കിൽ (visited.find(tensor) != visited.end()) {
        തിരികെ;
    }
    visited.insert(tensor);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

ശൂന്യം apir_serialize_graph(സ്ഥിരം ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    വേണ്ടി (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    പൂർണ്ണസംഖ്യ      output_size =
        വലുപ്പം(uint32_t) + n_nodes * വലുപ്പം(uint64_t) + വലുപ്പം(uint32_t) + n_tensors * വലുപ്പം(apir_rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, വലുപ്പം(n_nodes));
    വേണ്ടി (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + വലുപ്പം(n_nodes) + i * വലുപ്പം(uint64_t), &cgraph->nodes[i], വലുപ്പം(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + വലുപ്പം(n_nodes) + n_nodes * വലുപ്പം(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.data() + വലുപ്പം(n_nodes) + n_nodes * വലുപ്പം(uint64_t) + വലുപ്പം(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * വലുപ്പം(apir_rpc_tensor));
}

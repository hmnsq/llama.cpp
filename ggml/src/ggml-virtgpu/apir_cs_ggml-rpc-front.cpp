#அந்தர்பாவ "backend/shared/apir_cs_rpc.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-remoting.h"

#அந்தர்பாவ <cinttypes>
#அந்தர்பாவ <unordered_map>
#அந்தர்பாவ <unordered_set>
#அந்தர்பாவ <vector>

apir_rpc_tensor apir_serialize_tensor(மாறிலி ggml_tensor * tensor) {
    apir_rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    எனில் (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;

        result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
    } இல்லையெனில் {
        result.buffer = 0;
    }
    க்கு (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    க்கு (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / அளவு(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    க்கு (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    எனில் (tensor->data) {
        எனில் (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    திரும்பு result;
}

வெறுமை apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    எனில் (tensor == nullptr) {
        திரும்பு;
    }
    எனில் (visited.find(tensor) != visited.end()) {
        திரும்பு;
    }
    visited.insert(tensor);
    க்கு (முழு i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

வெறுமை apir_serialize_graph(மாறிலி ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    க்கு (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    முழு      output_size =
        அளவு(uint32_t) + n_nodes * அளவு(uint64_t) + அளவு(uint32_t) + n_tensors * அளவு(apir_rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, அளவு(n_nodes));
    க்கு (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + அளவு(n_nodes) + i * அளவு(uint64_t), &cgraph->nodes[i], அளவு(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + அளவு(n_nodes) + n_nodes * அளவு(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.data() + அளவு(n_nodes) + n_nodes * அளவு(uint64_t) + அளவு(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * அளவு(apir_rpc_tensor));
}

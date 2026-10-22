#ଅନ୍ତର୍ଭାଵ "backend/shared/apir_cs_rpc.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-remoting.h"

#ଅନ୍ତର୍ଭାଵ <cinttypes>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <unordered_set>
#ଅନ୍ତର୍ଭାଵ <vector>

apir_rpc_tensor apir_serialize_tensor(ଅଚଳ ggml_tensor * tensor) {
    apir_rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    ଯଦି (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;

        result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
    } ଅନ୍ୟଥା {
        result.buffer = 0;
    }
    ପାଇଁ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    ପାଇଁ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ଆକାର(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    ପାଇଁ (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    ଯଦି (tensor->data) {
        ଯଦି (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    ଫେରନ୍ତୁ result;
}

ଶୂନ୍ୟ apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    ଯଦି (tensor == nullptr) {
        ଫେରନ୍ତୁ;
    }
    ଯଦି (visited.find(tensor) != visited.end()) {
        ଫେରନ୍ତୁ;
    }
    visited.insert(tensor);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

ଶୂନ୍ୟ apir_serialize_graph(ଅଚଳ ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    ପାଇଁ (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      output_size =
        ଆକାର(uint32_t) + n_nodes * ଆକାର(uint64_t) + ଆକାର(uint32_t) + n_tensors * ଆକାର(apir_rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, ଆକାର(n_nodes));
    ପାଇଁ (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + ଆକାର(n_nodes) + i * ଆକାର(uint64_t), &cgraph->nodes[i], ଆକାର(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + ଆକାର(n_nodes) + n_nodes * ଆକାର(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.data() + ଆକାର(n_nodes) + n_nodes * ଆକାର(uint64_t) + ଆକାର(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * ଆକାର(apir_rpc_tensor));
}

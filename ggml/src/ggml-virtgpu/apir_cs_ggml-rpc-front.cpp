#include "backend/shared/apir_cs_rpc.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-remoting.h"

#include <cinttypes>
#include <unordered_map>
#include <unordered_set>
#include <vector>

apir_rpc_tensor apir_serialize_tensor(const ggml_tensor * tensor) {
    apir_rpc_tensor result;
    result.id   = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    ប្រសិនបើ (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;

        result.buffer = BUFFER_TO_HOST_HANDLE(buffer);
    } បើមិនដូច្នេះ {
        result.buffer = 0;
    }
    សម្រាប់ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    សម្រាប់ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / parimaa_na(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    សម្រាប់ (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src  = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data      = reinterpret_cast<uint64_t>(tensor->data);
    ប្រសិនបើ (tensor->data) {
        ប្រសិនបើ (!tensor->buffer) {
            GGML_ABORT("%s: tensor has data but not buffer", __func__);
        }
        // tensor->data is serialized as an offset to the buffer base address
        result.data -= reinterpret_cast<uint64_t>(BUFFER_TO_GGML_CONTEXT(tensor->buffer)->base);
    }
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    ត្រឡប់ result;
}

void apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited) {
    ប្រសិនបើ (tensor == nullptr) {
        ត្រឡប់;
    }
    ប្រសិនបើ (visited.find(tensor) != visited.end()) {
        ត្រឡប់;
    }
    visited.insert(tensor);
    សម្រាប់ (ចំនួនគត់ i = 0; i < GGML_MAX_SRC; i++) {
        apir_add_tensor(tensor->src[i], tensors, visited);
    }
    apir_add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(apir_serialize_tensor(tensor));
}

void apir_serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t                          n_nodes = cgraph->n_nodes;
    std::vector<apir_rpc_tensor>      tensors;
    std::unordered_set<ggml_tensor *> visited;
    សម្រាប់ (uint32_t i = 0; i < n_nodes; i++) {
        apir_add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(apir_rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    ចំនួនគត់      output_size =
        parimaa_na(uint32_t) + n_nodes * parimaa_na(uint64_t) + parimaa_na(uint32_t) + n_tensors * parimaa_na(apir_rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, parimaa_na(n_nodes));
    សម្រាប់ (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + parimaa_na(n_nodes) + i * parimaa_na(uint64_t), &cgraph->nodes[i], parimaa_na(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *) (output.data() + parimaa_na(n_nodes) + n_nodes * parimaa_na(uint64_t));
    *out_ntensors           = n_tensors;
    apir_rpc_tensor * out_tensors =
        (apir_rpc_tensor *) (output.data() + parimaa_na(n_nodes) + n_nodes * parimaa_na(uint64_t) + parimaa_na(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * parimaa_na(apir_rpc_tensor));
}

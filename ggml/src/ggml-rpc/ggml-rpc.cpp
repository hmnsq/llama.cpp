#ಅನ್ತರ್ಭಾವ "ggml-rpc.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpp.h"
#ಅನ್ತರ್ಭಾವ "transport.h"
#ಅನ್ತರ್ಭಾವ "log.h"

#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <chrono>
#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <thread>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <queue>
#ಅನ್ತರ್ಭಾವ <condition_variable>
#ಅನ್ತರ್ಭಾವ <future>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <filesystem>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <atomic>

ನಾಮಾಕಾಶ fs = std::filesystem;

// macro for nicer error messages on server crash
#ವ್ಯಾಖ್ಯಾನಿಸು RPC_ಸ್ಥಿತಿ_ASSERT(x) ಆದರೆ (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#ಸೂಚನೆ pack(ತಳ್ಳು, 1)
// ggml_tensor is serialized into rpc_tensor
ರಚನೆ rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t ಬಫರ್;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t ದತ್ತಾಂಶ;
    ಅಕ್ಷರ name[GGML_MAX_NAME];

    int32_t use_ಎಣಿಕೆ;
};

static_assert(ಗಾತ್ರ(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
ಎಣಿಕೆ rpc_cmd {
    RPC_CMD_ALLOC_ಬಫರ್ = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_ಗಾತ್ರ,
    RPC_CMD_ಬಫರ್_GET_ತಳ,
    RPC_CMD_FREE_ಬಫರ್,
    RPC_CMD_ಬಫರ್_CLEAR,
    RPC_CMD_ಗಣ_TENSOR,
    RPC_CMD_ಗಣ_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_ನಕ್ಷೆ_COMPUTE,
    RPC_CMD_GET_DEVICE_ಸ್ಮರಣೆ,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_ಗಾತ್ರ,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_ಎಣಿಕೆ,
    RPC_CMD_ನಕ್ಷೆ_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_ಗಣ_TENSOR_2_D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_ಎಣಿಕೆ,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * rpc_cmd_name(ಎಣಿಕೆ rpc_cmd cmd) {
    ಆಯ್ಕೆ (cmd) {
        ಸ್ಥಿತಿ RPC_CMD_ALLOC_ಬಫರ್:      ಹಿಂತಿರುಗಿ "ALLOC_BUFFER";
        ಸ್ಥಿತಿ RPC_CMD_GET_ALIGNMENT:     ಹಿಂತಿರುಗಿ "GET_ALIGNMENT";
        ಸ್ಥಿತಿ RPC_CMD_GET_MAX_ಗಾತ್ರ:      ಹಿಂತಿರುಗಿ "GET_MAX_SIZE";
        ಸ್ಥಿತಿ RPC_CMD_ಬಫರ್_GET_ತಳ:   ಹಿಂತಿರುಗಿ "BUFFER_GET_BASE";
        ಸ್ಥಿತಿ RPC_CMD_FREE_ಬಫರ್:       ಹಿಂತಿರುಗಿ "FREE_BUFFER";
        ಸ್ಥಿತಿ RPC_CMD_ಬಫರ್_CLEAR:      ಹಿಂತಿರುಗಿ "BUFFER_CLEAR";
        ಸ್ಥಿತಿ RPC_CMD_ಗಣ_TENSOR:        ಹಿಂತಿರುಗಿ "SET_TENSOR";
        ಸ್ಥಿತಿ RPC_CMD_ಗಣ_TENSOR_HASH:   ಹಿಂತಿರುಗಿ "SET_TENSOR_HASH";
        ಸ್ಥಿತಿ RPC_CMD_GET_TENSOR:        ಹಿಂತಿರುಗಿ "GET_TENSOR";
        ಸ್ಥಿತಿ RPC_CMD_COPY_TENSOR:       ಹಿಂತಿರುಗಿ "COPY_TENSOR";
        ಸ್ಥಿತಿ RPC_CMD_ನಕ್ಷೆ_COMPUTE:     ಹಿಂತಿರುಗಿ "GRAPH_COMPUTE";
        ಸ್ಥಿತಿ RPC_CMD_GET_DEVICE_ಸ್ಮರಣೆ: ಹಿಂತಿರುಗಿ "GET_DEVICE_MEMORY";
        ಸ್ಥಿತಿ RPC_CMD_INIT_TENSOR:       ಹಿಂತಿರುಗಿ "INIT_TENSOR";
        ಸ್ಥಿತಿ RPC_CMD_GET_ALLOC_ಗಾತ್ರ:    ಹಿಂತಿರುಗಿ "GET_ALLOC_SIZE";
        ಸ್ಥಿತಿ RPC_CMD_HELLO:             ಹಿಂತಿರುಗಿ "HELLO";
        ಸ್ಥಿತಿ RPC_CMD_DEVICE_ಎಣಿಕೆ:      ಹಿಂತಿರುಗಿ "DEVICE_COUNT";
        ಸ್ಥಿತಿ RPC_CMD_ನಕ್ಷೆ_RECOMPUTE:   ಹಿಂತಿರುಗಿ "GRAPH_RECOMPUTE";
        ಸ್ಥಿತಿ RPC_CMD_MEMSET_TENSOR:     ಹಿಂತಿರುಗಿ "MEMSET_TENSOR";
        ಸ್ಥಿತಿ RPC_CMD_ಗಣ_TENSOR_2_D:     ಹಿಂತಿರುಗಿ "SET_TENSOR_2D";
        ಸ್ಥಿತಿ RPC_CMD_GET_TENSOR_2D:     ಹಿಂತಿರುಗಿ "GET_TENSOR_2D";
        ಸ್ಥಿತಿ RPC_CMD_COMM_INIT:         ಹಿಂತಿರುಗಿ "COMM_INIT";
        ಸ್ಥಿತಿ RPC_CMD_COMM_ALLREDUCE:    ಹಿಂತಿರುಗಿ "COMM_ALLREDUCE";
        ಸ್ಥಿತಿ RPC_CMD_COMM_FREE:         ಹಿಂತಿರುಗಿ "COMM_FREE";
        ಸ್ಥಿತಿ RPC_CMD_NONE:              ಹಿಂತಿರುಗಿ "NONE";
        ಪೂರ್ವನಿಯೋಜಿತ:                        ಹಿಂತಿರುಗಿ "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ಸ್ಥಿರಾಂಕ size_t HASH_ಮಿತಿ = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ಸ್ಥಿರಾಂಕ size_t ನಕ್ಷೆ_CACHE_MAX = 1024;

ರಚನೆ rpc_ಸಂದೇಶ_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_ಗಾತ್ರ];
};

ರಚನೆ rpc_ಸಂದೇಶ_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_ಗಾತ್ರ];
};

ರಚನೆ rpc_ಸಂದೇಶ_device_ಎಣಿಕೆ_rsp {
    uint32_t device_ಎಣಿಕೆ;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

ರಚನೆ rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_rsp {
    uint64_t alloc_ಗಾತ್ರ;
};

ರಚನೆ rpc_ಸಂದೇಶ_init_tensor_req {
    rpc_tensor tensor;
};

ರಚನೆ rpc_ಸಂದೇಶ_alloc_ಬಫರ್_req {
    uint32_t device;
    uint64_t size;
};

ರಚನೆ rpc_ಸಂದೇಶ_alloc_ಬಫರ್_rsp {
    uint64_t remote_ptr;
    uint64_t remote_ಗಾತ್ರ;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_alignment_req {
    uint32_t device;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_alignment_rsp {
    uint64_t alignment;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_req {
    uint32_t device;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_rsp {
    uint64_t max_size;
};

ರಚನೆ rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_req {
    uint64_t remote_ptr;
};

ರಚನೆ rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_rsp {
    uint64_t ತಳ_ptr;
};

ರಚನೆ rpc_ಸಂದೇಶ_free_ಬಫರ್_req {
    uint64_t remote_ptr;
};

ರಚನೆ rpc_ಸಂದೇಶ_ಬಫರ್_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

ರಚನೆ rpc_ಸಂದೇಶ_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t ಸ್ಥಾನಾಂತರ;
    uint64_t size;
    uint8_t value;
};

ರಚನೆ rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t ಸ್ಥಾನಾಂತರ;
    uint64_t hash;
};

ರಚನೆ rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_rsp {
    uint8_t ಫಲಿತಾಂಶ;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_tensor_req {
    rpc_tensor tensor;
    uint64_t ಸ್ಥಾನಾಂತರ;
    uint64_t size;
};

ರಚನೆ rpc_ಸಂದೇಶ_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

ರಚನೆ rpc_ಸಂದೇಶ_copy_tensor_rsp {
    uint8_t ಫಲಿತಾಂಶ;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_req {
    uint32_t device;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_rsp {
    uint64_t free_mem;
    uint64_t ಒಟ್ಟು_mem;
};

ರಚನೆ rpc_ಸಂದೇಶ_ನಕ್ಷೆ_recompute_req {
    uint32_t device;
    uint64_t uid;
};

ರಚನೆ rpc_ಸಂದೇಶ_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t ಸ್ಥಾನಾಂತರ;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

ರಚನೆ rpc_ಸಂದೇಶ_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t ಪೋರ್ಟ್;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    ಅಕ್ಷರ     host[64];  // rank > 0: rank 0's host
};

ರಚನೆ rpc_ಸಂದೇಶ_comm_init_rsp {
    uint8_t ok;
};

ರಚನೆ rpc_ಸಂದೇಶ_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

ರಚನೆ rpc_ಸಂದೇಶ_comm_free_req {
    uint32_t device;
};

#ಸೂಚನೆ pack(ತೆಗೆ)

// RPC data structures

ಸ್ಥಿರ ggml_guid_t ggml_backend_rpc_guid() {
    ಸ್ಥಿರ ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    ಹಿಂತಿರುಗಿ &guid;
}

ರಚನೆ ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

ರಚನೆ ggml_backend_rpc_ಬಫರ್_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

ವರ್ಗ rpc_dispatcher;
ರಚನೆ ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

ರಚನೆ ggml_backend_rpc_ಬಫರ್_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    ಶೂನ್ಯ                            * ತಳ_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
ಸ್ಥಿರ uint64_t fnv_hash(ಸ್ಥಿರಾಂಕ uint8_t * ದತ್ತಾಂಶ, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ಸ್ಥಿರಾಂಕ uint64_t fnv_ಅವಿಭಾಜ್ಯ = 0x100000001b3ULL;

    ಗಾಗಿ (size_t i = 0; i < len; ++i) {
        hash ^= ದತ್ತಾಂಶ[i];
        hash *= fnv_ಅವಿಭಾಜ್ಯ;
    }
    ಹಿಂತಿರುಗಿ hash;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ send_ಸಂದೇಶ(socket_ptr sock, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಸಂದೇಶ_2, size_t ಸಂದೇಶ_ಗಾತ್ರ) {
    ಆದರೆ (!sock->send_ದತ್ತಾಂಶ(&ಸಂದೇಶ_ಗಾತ್ರ, ಗಾತ್ರ(ಸಂದೇಶ_ಗಾತ್ರ))) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!sock->send_ದತ್ತಾಂಶ(ಸಂದೇಶ_2, ಸಂದೇಶ_ಗಾತ್ರ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ sock->flush();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ recv_ಸಂದೇಶ(socket_ptr sock, ಶೂನ್ಯ * ಸಂದೇಶ_2, size_t ಸಂದೇಶ_ಗಾತ್ರ) {
    uint64_t size;
    ಆದರೆ (!sock->recv_ದತ್ತಾಂಶ(&size, ಗಾತ್ರ(size))) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (size != ಸಂದೇಶ_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ sock->recv_ದತ್ತಾಂಶ(ಸಂದೇಶ_2, ಸಂದೇಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ recv_ಸಂದೇಶ(socket_ptr sock, std::vector<uint8_t> & ಒಳಹರಿವು) {
    uint64_t size;
    ಆದರೆ (!sock->recv_ದತ್ತಾಂಶ(&size, ಗಾತ್ರ(size))) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಪ್ರಯತ್ನಿಸು {
        ಒಳಹರಿವು.resize(size);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ sock->recv_ದತ್ತಾಂಶ(ಒಳಹರಿವು.ದತ್ತಾಂಶ(), size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ parse_endpoint(ಸ್ಥಿರಾಂಕ std::string & endpoint, std::string & host, ಪೂರ್ಣಾಂಕ & ಪೋರ್ಟ್) {
    size_t pos = endpoint.find(':');
    ಆದರೆ (pos == std::string::npos) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    host = endpoint.substr(0, pos);
    ಪ್ರಯತ್ನಿಸು {
        ಪೋರ್ಟ್ = std::stoi(endpoint.substr(pos + 1));
    } ಹಿಡಿ (...) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
ಸ್ಥಿರ ತಾರ್ಕಿಕ send_rpc_cmd(socket_ptr sock, ಎಣಿಕೆ rpc_cmd cmd, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ) {
    uint8_t cmd_byte = cmd;
    ಆದರೆ (!sock->send_ದತ್ತಾಂಶ(&cmd_byte, ಗಾತ್ರ(cmd_byte))) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!sock->send_ದತ್ತಾಂಶ(&ಒಳಹರಿವು_ಗಾತ್ರ, ಗಾತ್ರ(ಒಳಹರಿವು_ಗಾತ್ರ))) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!sock->send_ದತ್ತಾಂಶ(ಒಳಹರಿವು, ಒಳಹರಿವು_ಗಾತ್ರ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
ಸ್ಥಿರ ತಾರ್ಕಿಕ send_rpc_cmd(socket_ptr sock, ಎಣಿಕೆ rpc_cmd cmd, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ, ಶೂನ್ಯ * output, size_t output_ಗಾತ್ರ) {
    ಆದರೆ (!send_rpc_cmd(sock, cmd, ಒಳಹರಿವು, ಒಳಹರಿವು_ಗಾತ್ರ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint64_t ಹೊರಹೋಗು_ಗಾತ್ರ;
    ಆದರೆ (!sock->recv_ದತ್ತಾಂಶ(&ಹೊರಹೋಗು_ಗಾತ್ರ, ಗಾತ್ರ(ಹೊರಹೋಗು_ಗಾತ್ರ))) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ಹೊರಹೋಗು_ಗಾತ್ರ != output_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!sock->recv_ದತ್ತಾಂಶ(output, output_ಗಾತ್ರ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
ಸ್ಥಿರ constexpr ಸ್ವಯಂಚಾಲಿತ RPC_BUSY_SPIN_IDLE_ಸಮಯ = std::chrono::milliseconds(100);

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rpc_cpu_relax() {
#ಆದರೆ defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ ಅಸ್ಥಿರ("yield" ::: "memory");
#ಇಲ್ಲದಿದ್ದರೆ
    std::this_thread::yield();
#ಆದರೆ_ಅಂತ್ಯ
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
ಸ್ಥಿರ ತಾರ್ಕಿಕ negotiate_hello(ಸ್ಥಿರಾಂಕ std::shared_ptr<socket_t> & sock) {
    rpc_ಸಂದೇಶ_hello_req request = {};
    rpc_ಸಂದೇಶ_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    ತಾರ್ಕಿಕ ಸ್ಥಿತಿ_2 = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, ಗಾತ್ರ(request), &response, ಗಾತ್ರ(response));
    RPC_ಸ್ಥಿತಿ_ASSERT(ಸ್ಥಿತಿ_2);

    ಆದರೆ (response.major != RPC_PROTO_MAJOR_ಆವೃತ್ತಿ || response.minor > RPC_PROTO_MINOR_ಆವೃತ್ತಿ) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ವರ್ಗ ಸಂದೇಶ_ಸರತಿ {
ಸಾರ್ವಜನಿಕ:
    ಸಂದೇಶ_ಸರತಿ() {}

    ತಾರ್ಕಿಕ ತಳ್ಳು(ಸ್ಥಿರಾಂಕ T &value) {
        std::unique_lock<std::mutex> ಬೀಗ(mutex);
        ಆದರೆ (interrupted) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        queue.ತಳ್ಳು(value);
        cvar.notify_all();
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ ತೆಗೆ(T* ಹೊರಹೋಗು) {
        std::unique_lock<std::mutex> ಬೀಗ(mutex);
        cvar.wait(ಬೀಗ, [ಇದು] { ಹಿಂತಿರುಗಿ !queue.ಖಾಲಿಗಳು() || interrupted; });
        ಆದರೆ (interrupted) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        *ಹೊರಹೋಗು = queue.front();
        queue.ತೆಗೆ();
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ try_ತೆಗೆ(T* ಹೊರಹೋಗು) {
        std::unique_lock<std::mutex> ಬೀಗ(mutex);
        ಆದರೆ (interrupted || queue.ಖಾಲಿಗಳು()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        *ಹೊರಹೋಗು = queue.front();
        queue.ತೆಗೆ();
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಶೂನ್ಯ ಅಡಚಣೆ() {
        std::unique_lock<std::mutex> ಬೀಗ(mutex);
        interrupted = ನಿಜ;
        ಬೀಗ.unlock();
        cvar.notify_all();
    }

ಖಾಸಗಿ:
    ತಾರ್ಕಿಕ interrupted = ಸುಳ್ಳು;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

ವರ್ಗ rpc_dispatcher {
ಸಾರ್ವಜನಿಕ:
    rpc_dispatcher() {
    }

    ಶೂನ್ಯ send(ಎಣಿಕೆ rpc_cmd cmd, std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ);
    ಶೂನ್ಯ send(ಎಣಿಕೆ rpc_cmd cmd, std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ, ಶೂನ್ಯ * output, size_t output_ಗಾತ್ರ);
    ಶೂನ್ಯ send_async(ಎಣಿಕೆ rpc_cmd cmd, std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ);
    ಶೂನ್ಯ send_async(ಎಣಿಕೆ rpc_cmd cmd, std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ, ಶೂನ್ಯ * output, size_t output_ಗಾತ್ರ);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    ಶೂನ್ಯ event_free(ggml_backend_event_t event);
    ಶೂನ್ಯ event_synchronize(ggml_backend_event_t event);
    ಶೂನ್ಯ event_ದಾಖಲೆ(ggml_backend_event_t event);
    ಶೂನ್ಯ synchronize();
    ಶೂನ್ಯ busy_spin_acquire();
    ಶೂನ್ಯ busy_spin_release();
    ಶೂನ್ಯ ನಕ್ಷೆ_compute(uint32_t device, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph);

    ಶೂನ್ಯ start(ಸ್ಥಿರಾಂಕ std::string & endpoint);
    ಶೂನ್ಯ work();

    ~rpc_dispatcher();

ಖಾಸಗಿ:
    ರಚನೆ rpc_ಸಂದೇಶ {
        rpc_cmd                       cmd;
        std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ>   ಒಳಹರಿವು;
        size_t                        ಒಳಹರಿವು_ಗಾತ್ರ;
        ಶೂನ್ಯ                        * output;
        size_t                        output_ಗಾತ್ರ;
        std::promise<ಶೂನ್ಯ>            completion;
    };
    ಬಳಸಿ rpc_ಸಂದೇಶ_ptr   = std::shared_ptr<rpc_ಸಂದೇಶ>;
    ಬಳಸಿ rpc_ಸಂದೇಶ_ಸರತಿ = ಸಂದೇಶ_ಸರತಿ<rpc_ಸಂದೇಶ_ptr>;
    ರಚನೆ rpc_event {
        rpc_ಸಂದೇಶ_ptr              ಸಂದೇಶ_2;
        std::shared_future<ಶೂನ್ಯ> sf;
    };
    std::mutex ನಕ್ಷೆ_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> ನಕ್ಷೆ_uids;
    rpc_ಸಂದೇಶ_ಸರತಿ    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

ಸ್ಥಿರ ಶೂನ್ಯ rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

ಶೂನ್ಯ rpc_dispatcher::send(ಎಣಿಕೆ rpc_cmd cmd, std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ) {
    ಸ್ವಯಂಚಾಲಿತ ಸಂದೇಶ_2 = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ_2->cmd = cmd;
    ಸಂದೇಶ_2->ಒಳಹರಿವು = ಒಳಹರಿವು;
    ಸಂದೇಶ_2->ಒಳಹರಿವು_ಗಾತ್ರ = ಒಳಹರಿವು_ಗಾತ್ರ;
    ಸಂದೇಶ_2->output = ಶೂನ್ಯಸೂಚಕ;
    ಸಂದೇಶ_2->output_ಗಾತ್ರ = 0;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ_2));
    ಸ್ವಯಂಚಾಲಿತ future = ಸಂದೇಶ_2->completion.get_future();
    future.wait();
}

ಶೂನ್ಯ rpc_dispatcher::send_async(ಎಣಿಕೆ rpc_cmd cmd, std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ) {
    ಸ್ವಯಂಚಾಲಿತ ಸಂದೇಶ_2 = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ_2->cmd = cmd;
    ಸಂದೇಶ_2->ಒಳಹರಿವು = ಒಳಹರಿವು;
    ಸಂದೇಶ_2->ಒಳಹರಿವು_ಗಾತ್ರ = ಒಳಹರಿವು_ಗಾತ್ರ;
    ಸಂದೇಶ_2->output = ಶೂನ್ಯಸೂಚಕ;
    ಸಂದೇಶ_2->output_ಗಾತ್ರ = 0;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ_2));
}

ಶೂನ್ಯ rpc_dispatcher::send(ಎಣಿಕೆ rpc_cmd cmd, std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ, ಶೂನ್ಯ * output, size_t output_ಗಾತ್ರ) {
    ಸ್ವಯಂಚಾಲಿತ ಸಂದೇಶ_2 = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ_2->cmd = cmd;
    ಸಂದೇಶ_2->ಒಳಹರಿವು = ಒಳಹರಿವು;
    ಸಂದೇಶ_2->ಒಳಹರಿವು_ಗಾತ್ರ = ಒಳಹರಿವು_ಗಾತ್ರ;
    ಸಂದೇಶ_2->output = output;
    ಸಂದೇಶ_2->output_ಗಾತ್ರ = output_ಗಾತ್ರ;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ_2));
    ಸ್ವಯಂಚಾಲಿತ future = ಸಂದೇಶ_2->completion.get_future();
    future.wait();
}

ಶೂನ್ಯ rpc_dispatcher::send_async(ಎಣಿಕೆ rpc_cmd cmd, std::shared_ptr<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ> ಒಳಹರಿವು, size_t ಒಳಹರಿವು_ಗಾತ್ರ, ಶೂನ್ಯ * output, size_t output_ಗಾತ್ರ) {
    ಸ್ವಯಂಚಾಲಿತ ಸಂದೇಶ_2 = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ_2->cmd = cmd;
    ಸಂದೇಶ_2->ಒಳಹರಿವು = ಒಳಹರಿವು;
    ಸಂದೇಶ_2->ಒಳಹರಿವು_ಗಾತ್ರ = ಒಳಹರಿವು_ಗಾತ್ರ;
    ಸಂದೇಶ_2->output = output;
    ಸಂದೇಶ_2->output_ಗಾತ್ರ = output_ಗಾತ್ರ;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ_2));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = ಹೊಸ rpc_event;
    ev->ಸಂದೇಶ_2 = std::make_shared<rpc_ಸಂದೇಶ>();
    ev->ಸಂದೇಶ_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->ಸಂದೇಶ_2->completion.get_future().share();
    GGML_ASSERT(queue.ತಳ್ಳು(ev->ಸಂದೇಶ_2));
    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

ಶೂನ್ಯ rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ಅಳಿಸು ev;
}

ಶೂನ್ಯ rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

ಶೂನ್ಯ rpc_dispatcher::event_ದಾಖಲೆ(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->ಸಂದೇಶ_2 = std::make_shared<rpc_ಸಂದೇಶ>();
    ev->ಸಂದೇಶ_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->ಸಂದೇಶ_2->completion.get_future().share();
    GGML_ASSERT(queue.ತಳ್ಳು(ev->ಸಂದೇಶ_2));
}

ಶೂನ್ಯ rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    ಸ್ವಯಂಚಾಲಿತ ಸಂದೇಶ_2 = std::make_shared<rpc_ಸಂದೇಶ>();
    ಸಂದೇಶ_2->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ತಳ್ಳು(ಸಂದೇಶ_2));
    ಸಂದೇಶ_2->completion.get_future().wait();
}

ಶೂನ್ಯ rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

ಶೂನ್ಯ rpc_dispatcher::busy_spin_release() {
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಹಿಂದಿನ = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(ಹಿಂದಿನ > 0);
}

ಶೂನ್ಯ rpc_dispatcher::start(ಸ್ಥಿರಾಂಕ std::string & endpoint) {
    std::string host;
    ಪೂರ್ಣಾಂಕ ಪೋರ್ಟ್;
    ಆದರೆ (!parse_endpoint(endpoint, host, ಪೋರ್ಟ್)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    ಆದರೆ (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), ಪೋರ್ಟ್);
    ಆದರೆ (sock == ಶೂನ್ಯಸೂಚಕ) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    ಆದರೆ (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = ನಿಜ;
    thread = std::thread(rpc_dispatcher_trampoline, ಇದು);
}

ಶೂನ್ಯ rpc_dispatcher::work() {
    ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_cmd = std::chrono::steady_clock::now();
    ಆಗ (running) {
        rpc_ಸಂದೇಶ_ptr ಸಂದೇಶ_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - ಕೊನೆಯ_cmd < RPC_BUSY_SPIN_IDLE_ಸಮಯ;
        ಆದರೆ (spin) {
            ಆದರೆ (!queue.try_ತೆಗೆ(&ಸಂದೇಶ_ptr)) {
                rpc_cpu_relax();
                ಮುಂದುವರಿ;
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!queue.ತೆಗೆ(&ಸಂದೇಶ_ptr)) {
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (ಸಂದೇಶ_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(ಸಂದೇಶ_ptr->cmd),
                     rpc_log_format_ಗಾತ್ರ(ಸಂದೇಶ_ptr->ಒಳಹರಿವು_ಗಾತ್ರ).c_str(), rpc_log_format_ಗಾತ್ರ(ಸಂದೇಶ_ptr->output_ಗಾತ್ರ).c_str());
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ trace = rpc_log_debug_level() >= 3;
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ t0 = std::chrono::steady_clock::now();
            ಆದರೆ (ಸಂದೇಶ_ptr->output) {
                ತಾರ್ಕಿಕ ಸ್ಥಿತಿ_2 = send_rpc_cmd(sock, ಸಂದೇಶ_ptr->cmd, ಸಂದೇಶ_ptr->ಒಳಹರಿವು.get(), ಸಂದೇಶ_ptr->ಒಳಹರಿವು_ಗಾತ್ರ, ಸಂದೇಶ_ptr->output, ಸಂದೇಶ_ptr->output_ಗಾತ್ರ);
                RPC_ಸ್ಥಿತಿ_ASSERT(ಸ್ಥಿತಿ_2);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ತಾರ್ಕಿಕ ಸ್ಥಿತಿ_2 = send_rpc_cmd(sock, ಸಂದೇಶ_ptr->cmd, ಸಂದೇಶ_ptr->ಒಳಹರಿವು.get(), ಸಂದೇಶ_ptr->ಒಳಹರಿವು_ಗಾತ್ರ);
                RPC_ಸ್ಥಿತಿ_ASSERT(ಸ್ಥಿತಿ_2);
            }
            ಆದರೆ (trace) {
                ಸ್ಥಿರಾಂಕ std::chrono::duration<ದ್ವಿಗುಣ, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(ಸಂದೇಶ_ptr->cmd), elapsed.count());
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        ಸಂದೇಶ_ptr->completion.set_value();
        ಕೊನೆಯ_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = ಸುಳ್ಳು;
    queue.ಅಡಚಣೆ();
    sock = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (thread.joinable()) {
        thread.join();
    }
}

ಸ್ಥಿರ std::shared_ptr<rpc_dispatcher> get_dispatcher(ಸ್ಥಿರಾಂಕ std::string & endpoint) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);
    ಸ್ಥಿರ std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    ಸ್ವಯಂಚಾಲಿತ it = dispatchers.find(endpoint);
    ಆದರೆ (it != dispatchers.end()) {
        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ dispatcher = it->second.ಬೀಗ()) {
            ಹಿಂತಿರುಗಿ dispatcher;
        }
    }

    ಸ್ವಯಂಚಾಲಿತ dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    ಹಿಂತಿರುಗಿ dispatcher;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_free_ಬಫರ್_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_ಬಫರ್, request, ಗಾತ್ರ(*request));
    ಅಳಿಸು ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_rpc_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
    ಆದರೆ (ctx->ತಳ_ptr != ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ctx->ತಳ_ptr;
    }
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_rsp response;
    ctx->dispatcher->send(RPC_CMD_ಬಫರ್_GET_ತಳ, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));
    ctx->ತಳ_ptr = reinterpret_cast<ಶೂನ್ಯ *>(response.ತಳ_ptr);
    ಹಿಂತಿರುಗಿ ctx->ತಳ_ptr;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_rpc(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ ಬಫರ್->iface.free_ಬಫರ್ == ggml_backend_rpc_ಬಫರ್_free_ಬಫರ್;
}

ಸ್ಥಿರ rpc_tensor serialize_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ rpc_dispatcher * dispatcher = ಶೂನ್ಯಸೂಚಕ) {
    rpc_tensor ಫಲಿತಾಂಶ;
    ಆದರೆ (!tensor) {
        memset(&ಫಲಿತಾಂಶ, 0, ಗಾತ್ರ(ಫಲಿತಾಂಶ));
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    ಫಲಿತಾಂಶ.id = reinterpret_cast<uint64_t>(tensor);
    ಫಲಿತಾಂಶ.type = tensor->type;
    ಆದರೆ (tensor->ಬಫರ್ && ggml_backend_ಬಫರ್_is_rpc(tensor->ಬಫರ್)) {
        ggml_backend_ಬಫರ್_t ಬಫರ್ = tensor->ಬಫರ್;
        ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        ಆದರೆ (ctx != ಶೂನ್ಯಸೂಚಕ && (dispatcher == ಶೂನ್ಯಸೂಚಕ || ctx->dispatcher.get() == dispatcher)) {
            ಫಲಿತಾಂಶ.ಬಫರ್ = ctx->remote_ptr;
            ಫಲಿತಾಂಶ.ದತ್ತಾಂಶ = reinterpret_cast<uint64_t>(tensor->ದತ್ತಾಂಶ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಫಲಿತಾಂಶ.ಬಫರ್ = 0;
            ಫಲಿತಾಂಶ.ದತ್ತಾಂಶ = 0;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ.ಬಫರ್ = 0;
        ಫಲಿತಾಂಶ.ದತ್ತಾಂಶ   = 0;
    }
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ಫಲಿತಾಂಶ.ne[i] = tensor->ne[i];
        ಫಲಿತಾಂಶ.nb[i] = tensor->nb[i];
    }
    ಫಲಿತಾಂಶ.op = tensor->op;
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t); i++) {
        ಫಲಿತಾಂಶ.op_params[i] = tensor->op_params[i];
    }
    ಫಲಿತಾಂಶ.flags = tensor->flags;
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        ಫಲಿತಾಂಶ.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    ಫಲಿತಾಂಶ.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    ಫಲಿತಾಂಶ.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(ಫಲಿತಾಂಶ.name, 0, ಗಾತ್ರ(ಫಲಿತಾಂಶ.name));
    ಫಲಿತಾಂಶ.use_ಎಣಿಕೆ = 0;

    snprintf(ಫಲಿತಾಂಶ.name, GGML_MAX_NAME, "%s", tensor->name);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_rpc_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    ಆದರೆ (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == ಶೂನ್ಯಸೂಚಕ)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, ಗಾತ್ರ(*request));
    }
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಬಫರ್_memset_tensor(
        ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), ಸ್ಥಾನಾಂತರ,
             rpc_log_format_ಗಾತ್ರ(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, ಗಾತ್ರ(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
ಸ್ಥಿರ std::shared_ptr<uint8_t> serialize_ಗಣ_tensor(ಸ್ಥಿರಾಂಕ rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size, size_t & ಒಳಹರಿವು_ಗಾತ್ರ) {
    ಒಳಹರಿವು_ಗಾತ್ರ = ಗಾತ್ರ(rpc_tensor) + ಗಾತ್ರ(cache_flag) + ಗಾತ್ರ(ಸ್ಥಾನಾಂತರ) + size;
    uint8_t * ಒಳಹರಿವು = ಹೊಸ uint8_t[ಒಳಹರಿವು_ಗಾತ್ರ]();
    uint8_t * p = ಒಳಹರಿವು;
    ಸ್ಮರಣೆ_ನಕಲು(p, &rpc_tensor, ಗಾತ್ರ(rpc_tensor)); p += ಗಾತ್ರ(rpc_tensor);
    ಸ್ಮರಣೆ_ನಕಲು(p, &cache_flag, ಗಾತ್ರ(cache_flag)); p += ಗಾತ್ರ(cache_flag);
    ಸ್ಮರಣೆ_ನಕಲು(p, &ಸ್ಥಾನಾಂತರ,     ಗಾತ್ರ(ಸ್ಥಾನಾಂತರ));     p += ಗಾತ್ರ(ಸ್ಥಾನಾಂತರ);
    ಸ್ಮರಣೆ_ನಕಲು(p, ದತ್ತಾಂಶ, size);
    ಹಿಂತಿರುಗಿ std::shared_ptr<uint8_t>(ಒಳಹರಿವು, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
ಸ್ಥಿರ ತಾರ್ಕಿಕ rpc_use_hash_cache(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, size_t size) {
    ಹಿಂತಿರುಗಿ size > HASH_ಮಿತಿ && tensor->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), ಸ್ಥಾನಾಂತರ,
             rpc_log_format_ಗಾತ್ರ(size).c_str());
    ಆದರೆ (rpc_use_hash_cache(tensor, size)) {
        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ;
        request->hash = fnv_hash((ಸ್ಥಿರಾಂಕ uint8_t*)ದತ್ತಾಂಶ, size);
        rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_ಗಣ_TENSOR_HASH, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.ಫಲಿತಾಂಶ ? "hit" : "miss");
        ಆದರೆ (response.ಫಲಿತಾಂಶ) {
            // the server has the same data, no need to send it
            ಹಿಂತಿರುಗಿ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ಒಳಹರಿವು_ಗಾತ್ರ;
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = serialize_ಗಣ_tensor(rpc_tensor, cache_flag, ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size, ಒಳಹರಿವು_ಗಾತ್ರ);
    ctx->dispatcher->send(RPC_CMD_ಗಣ_TENSOR, ಒಳಹರಿವು, ಒಳಹರಿವು_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಬಫರ್_ಗಣ_tensor_2d(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ,
        size_t ಸ್ಥಾನಾಂತರ, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), ಸ್ಥಾನಾಂತರ, rpc_log_format_ಗಾತ್ರ(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t ಒಳಹರಿವು_ಗಾತ್ರ = ಗಾತ್ರ(rpc_tensor) + 4*ಗಾತ್ರ(uint64_t) + size*n_copies;
    uint8_t * ಒಳಹರಿವು = ಹೊಸ uint8_t[ಒಳಹರಿವು_ಗಾತ್ರ]();
    uint8_t * dest = ಒಳಹರಿವು;
    ಸ್ಮರಣೆ_ನಕಲು(dest, &rpc_tensor, ಗಾತ್ರ(rpc_tensor));
    dest += ಗಾತ್ರ(rpc_tensor);
    uint64_t ಶೀರ್ಷಿಕೆ[4] = { ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor };
    ಸ್ಮರಣೆ_ನಕಲು(dest, ಶೀರ್ಷಿಕೆ, ಗಾತ್ರ(ಶೀರ್ಷಿಕೆ));
    dest += ಗಾತ್ರ(ಶೀರ್ಷಿಕೆ);
    ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
        ಸ್ಮರಣೆ_ನಕಲು(dest + i*size, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)ದತ್ತಾಂಶ + i*stride_ದತ್ತಾಂಶ, size);
    }
    std::shared_ptr<uint8_t> ಒಳಹರಿವು_ptr(ಒಳಹರಿವು, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_ಗಣ_TENSOR_2_D, ಒಳಹರಿವು_ptr, ಒಳಹರಿವು_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಬಫರ್_get_tensor_2d(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ,
        size_t ಸ್ಥಾನಾಂತರ, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), ಸ್ಥಾನಾಂತರ, rpc_log_format_ಗಾತ್ರ(size).c_str(), n_copies, stride_tensor);
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->ಸ್ಥಾನಾಂತರ   = ಸ್ಥಾನಾಂತರ;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    ಆದರೆ (stride_ದತ್ತಾಂಶ == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, ಗಾತ್ರ(*request), ದತ್ತಾಂಶ, size*n_copies);
    } ಇಲ್ಲದಿದ್ದರೆ {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, ಗಾತ್ರ(*request), packed.ದತ್ತಾಂಶ(), packed.size());
        ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
            ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *)ದತ್ತಾಂಶ + i*stride_ದತ್ತಾಂಶ, packed.ದತ್ತಾಂಶ() + i*size, size);
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), ಸ್ಥಾನಾಂತರ,
             rpc_log_format_ಗಾತ್ರ(size).c_str());
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, ಗಾತ್ರ(*request), ದತ್ತಾಂಶ, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_rpc_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಆದರೆ (ggml_backend_ಬಫರ್_is_rpc(src->ಬಫರ್)) {
        // check if src and dst are on the same server
        ggml_backend_ಬಫರ್_t src_ಬಫರ್ = src->ಬಫರ್;
        ggml_backend_rpc_ಬಫರ್_context * src_ctx = (ggml_backend_rpc_ಬಫರ್_context *)src_ಬಫರ್->context;
        ggml_backend_ಬಫರ್_t dst_ಬಫರ್ = dst->ಬಫರ್;
        ggml_backend_rpc_ಬಫರ್_context * dst_ctx = (ggml_backend_rpc_ಬಫರ್_context *)dst_ಬಫರ್->context;
        ಆದರೆ (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_ಸಂದೇಶ_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));
        ಹಿಂತಿರುಗಿ response.ಫಲಿತಾಂಶ;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ggml_backend_rpc_ಬಫರ್_context * ctx = (ggml_backend_rpc_ಬಫರ್_context *)ಬಫರ್->context;
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_ಬಫರ್_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_ಬಫರ್_CLEAR, request, ಗಾತ್ರ(*request));
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_i ggml_backend_rpc_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_rpc_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_rpc_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_ಬಫರ್_ಗಣ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_ಬಫರ್_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_rpc_ಬಫರ್_type_name(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_rpc_ಬಫರ್_type_context * buft_ctx = (ggml_backend_rpc_ಬಫರ್_type_context *)buft->context;
    ಹಿಂತಿರುಗಿ buft_ctx->name.c_str();
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_rpc_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ggml_backend_rpc_ಬಫರ್_type_context * buft_ctx = (ggml_backend_rpc_ಬಫರ್_type_context *)buft->context;
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_alloc_ಬಫರ್_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_ಸಂದೇಶ_alloc_ಬಫರ್_rsp response;

    ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_ಬಫರ್, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));
    ಆದರೆ (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_ಗಾತ್ರ(request->size).c_str(),
                response.remote_ptr, rpc_log_format_ಗಾತ್ರ(response.remote_ಗಾತ್ರ).c_str());
        ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_ಬಫರ್_init(buft,
            ggml_backend_rpc_ಬಫರ್_interface,
            ಹೊಸ ggml_backend_rpc_ಬಫರ್_context{dispatcher, ಶೂನ್ಯಸೂಚಕ, response.remote_ptr},
            response.remote_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ಬಫರ್;
    } ಇಲ್ಲದಿದ್ದರೆ {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_ಗಾತ್ರ(request->size).c_str());
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}

ಸ್ಥಿರ size_t get_alignment(ಸ್ಥಿರಾಂಕ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_get_alignment_req>();
    request->device = device;
    rpc_ಸಂದೇಶ_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_ಗಾತ್ರ(response.alignment).c_str());
    ಹಿಂತಿರುಗಿ response.alignment;
}

ಸ್ಥಿರ size_t ggml_backend_rpc_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_rpc_ಬಫರ್_type_context * buft_ctx = (ggml_backend_rpc_ಬಫರ್_type_context *)buft->context;
    ಹಿಂತಿರುಗಿ buft_ctx->alignment;
}

ಸ್ಥಿರ size_t get_max_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_req>();
    request->device = device;
    rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_ಗಾತ್ರ, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_ಗಾತ್ರ(response.max_size).c_str());
    ಹಿಂತಿರುಗಿ response.max_size;
}

ಸ್ಥಿರ size_t ggml_backend_rpc_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_rpc_ಬಫರ್_type_context * buft_ctx = (ggml_backend_rpc_ಬಫರ್_type_context *)buft->context;
    ಹಿಂತಿರುಗಿ buft_ctx->max_size;
}

ಸ್ಥಿರ size_t ggml_backend_rpc_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    ತಾರ್ಕಿಕ rpc_get = ಸುಳ್ಳು;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == ಶೂನ್ಯಸೂಚಕ);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_ಗಾತ್ರ_may_expand(tensor->op);

    ಆದರೆ (rpc_get) {
        ggml_backend_rpc_ಬಫರ್_type_context * buft_ctx = (ggml_backend_rpc_ಬಫರ್_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ಸ್ಥಿರಾಂಕ size_t min_ಗಾತ್ರ = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        ರಚನೆ alloc_ಗಾತ್ರ_cache_ಕೀಲಿ {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_ಗಾತ್ರ_cache_ಕೀಲಿ ಕೀಲಿ = {};
        ಕೀಲಿ.device = buft_ctx->device;
        ಕೀಲಿ.type = tensor->type;
        ಕೀಲಿ.op = tensor->op;
        ಸ್ಮರಣೆ_ನಕಲು(ಕೀಲಿ.op_params, tensor->op_params, ಗಾತ್ರ(ಕೀಲಿ.op_params));
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
            ಕೀಲಿ.ne[i] = (uint32_t)tensor->ne[i];
            ಕೀಲಿ.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ಸ್ಥಿರಾಂಕ uint8_t *)&ಕೀಲಿ, ಗಾತ್ರ(ಕೀಲಿ));
        cache_hash = fnv_hash((ಸ್ಥಿರಾಂಕ uint8_t *)buft_ctx->endpoint.ದತ್ತಾಂಶ(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        ಸ್ಥಿರ std::mutex cache_mutex;
        ಸ್ಥಿರ std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> ಬೀಗ(cache_mutex);
            ಸ್ವಯಂಚಾಲಿತ it = cache.find(cache_hash);
            ಆದರೆ (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_ಗಾತ್ರ(it->second).c_str());
                ಹಿಂತಿರುಗಿ std::max<size_t>(it->second, min_ಗಾತ್ರ);
            }
        }

        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_rsp response;
        ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_ಗಾತ್ರ, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));

        {
            std::lock_guard<std::mutex> ಬೀಗ(cache_mutex);
            cache[cache_hash] = response.alloc_ಗಾತ್ರ;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_ಗಾತ್ರ(response.alloc_ಗಾತ್ರ).c_str());

        ಹಿಂತಿರುಗಿ std::max<size_t>(response.alloc_ಗಾತ್ರ, min_ಗಾತ್ರ);
    }

    ಹಿಂತಿರುಗಿ ggml_nbytes(tensor);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_i ggml_backend_rpc_ಬಫರ್_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_ಬಫರ್_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ಖಾಲಿ,
    /* .get_alignment       = */ ggml_backend_rpc_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_rpc_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    ಹಿಂತಿರುಗಿ rpc_ctx->name.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ಅಳಿಸು rpc_ctx;
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಗಣ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), ಸ್ಥಾನಾಂತರ,
             rpc_log_format_ಗಾತ್ರ(size).c_str());
    ಆದರೆ (rpc_use_hash_cache(tensor, size)) {
        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ;
        request->hash = fnv_hash((ಸ್ಥಿರಾಂಕ uint8_t*)ದತ್ತಾಂಶ, size);
        rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_ಗಣ_TENSOR_HASH, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.ಫಲಿತಾಂಶ ? "hit" : "miss");
        ಆದರೆ (response.ಫಲಿತಾಂಶ) {
            // the server has the same data, no need to send it
            ಹಿಂತಿರುಗಿ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ಒಳಹರಿವು_ಗಾತ್ರ;
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = serialize_ಗಣ_tensor(rpc_tensor, cache_flag, ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size, ಒಳಹರಿವು_ಗಾತ್ರ);
    ctx->dispatcher->send_async(RPC_CMD_ಗಣ_TENSOR, ಒಳಹರಿವು, ಒಳಹರಿವು_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), ಸ್ಥಾನಾಂತರ,
             rpc_log_format_ಗಾತ್ರ(size).c_str());
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->ಸ್ಥಾನಾಂತರ = ಸ್ಥಾನಾಂತರ;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, ಗಾತ್ರ(*request), ದತ್ತಾಂಶ, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

ಸ್ಥಿರ ಶೂನ್ಯ ಕೂಡಿಸು_tensor(ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & ಭೇಟಿಯಾದ) {
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (ಭೇಟಿಯಾದ.find(tensor) != ಭೇಟಿಯಾದ.end()) {
        ಹಿಂತಿರುಗಿ;
    }
    ಭೇಟಿಯಾದ.insert(tensor);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಕೂಡಿಸು_tensor(tensor->src[i], cgraph, dispatcher, tensors, ಭೇಟಿಯಾದ);
    }
    ಕೂಡಿಸು_tensor(tensor->view_src, cgraph, dispatcher, tensors, ಭೇಟಿಯಾದ);
    rpc_tensor ಫಲಿತಾಂಶ = serialize_tensor(tensor, dispatcher);
    ಸ್ಥಿರಾಂಕ size_t hash_pos = ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, tensor);
    ಆದರೆ (hash_pos != GGML_HASHSET_ತುಂಬಿದ && ggml_bitset_get(cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, hash_pos)) {
        ಫಲಿತಾಂಶ.use_ಎಣಿಕೆ = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(ಫಲಿತಾಂಶ);
}

ಸ್ಥಿರ uint8_t * serialize_ನಕ್ಷೆ(uint32_t device, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ rpc_dispatcher * dispatcher, size_t * output_ಗಾತ್ರ) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> ಭೇಟಿಯಾದ;
    ಗಾಗಿ (uint32_t i = 0; i < n_nodes; i++) {
        ಕೂಡಿಸು_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, ಭೇಟಿಯಾದ);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_ಗಾತ್ರ = 2*ಗಾತ್ರ(uint32_t) + ಗಾತ್ರ(uint64_t) + n_nodes * ಗಾತ್ರ(uint64_t) + ಗಾತ್ರ(uint32_t) + n_tensors * ಗಾತ್ರ(rpc_tensor);
    uint8_t * output = ಹೊಸ uint8_t[*output_ಗಾತ್ರ]();
    uint8_t * dest = output;
    ಸ್ಮರಣೆ_ನಕಲು(dest, &device, ಗಾತ್ರ(device));
    dest += ಗಾತ್ರ(device);
    ಸ್ಮರಣೆ_ನಕಲು(dest, &cgraph->uid, ಗಾತ್ರ(cgraph->uid));
    dest += ಗಾತ್ರ(cgraph->uid);
    ಸ್ಮರಣೆ_ನಕಲು(dest, &n_nodes, ಗಾತ್ರ(n_nodes));
    dest += ಗಾತ್ರ(n_nodes);
    ಗಾಗಿ (uint32_t i = 0; i < n_nodes; i++) {
        ಸ್ಮರಣೆ_ನಕಲು(dest + i * ಗಾತ್ರ(uint64_t), &cgraph->nodes[i], ಗಾತ್ರ(uint64_t));
    }
    dest += n_nodes * ಗಾತ್ರ(uint64_t);
    ಸ್ಮರಣೆ_ನಕಲು(dest, &n_tensors, ಗಾತ್ರ(n_tensors));
    dest += ಗಾತ್ರ(n_tensors);
    rpc_tensor * ಹೊರಹೋಗು_tensors = (rpc_tensor *)dest;
    ಸ್ಮರಣೆ_ನಕಲು(ಹೊರಹೋಗು_tensors, tensors.ದತ್ತಾಂಶ(), n_tensors * ಗಾತ್ರ(rpc_tensor));
    ಹಿಂತಿರುಗಿ output;
}

ಶೂನ್ಯ rpc_dispatcher::ನಕ್ಷೆ_compute(uint32_t device, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> ಬೀಗ(ನಕ್ಷೆ_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    ಸ್ವಯಂಚಾಲಿತ & device_ನಕ್ಷೆ_uids = ನಕ್ಷೆ_uids[device];
    ತಾರ್ಕಿಕ reuse = cgraph->uid != 0 && device_ನಕ್ಷೆ_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    ಆದರೆ (reuse) {
        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_ನಕ್ಷೆ_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_ನಕ್ಷೆ_RECOMPUTE, request, ಗಾತ್ರ(*request));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (cgraph->uid != 0) {
            ಆದರೆ (device_ನಕ್ಷೆ_uids.size() >= ನಕ್ಷೆ_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_ನಕ್ಷೆ_uids.size());
                device_ನಕ್ಷೆ_uids.clear();
            }
            device_ನಕ್ಷೆ_uids.insert(cgraph->uid);
        }
        size_t ಒಳಹರಿವು_ಗಾತ್ರ = 0;
        uint8_t * ಒಳಹರಿವು = serialize_ನಕ್ಷೆ(device, cgraph, ಇದು, &ಒಳಹರಿವು_ಗಾತ್ರ);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_ಗಾತ್ರ(ಒಳಹರಿವು_ಗಾತ್ರ).c_str());
        std::shared_ptr<uint8_t> ಒಳಹರಿವು_ptr(ಒಳಹರಿವು, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_ನಕ್ಷೆ_COMPUTE, ಒಳಹರಿವು_ptr, ಒಳಹರಿವು_ಗಾತ್ರ);
    }
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_rpc_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->ನಕ್ಷೆ_compute(rpc_ctx->device, cgraph);
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_event_ದಾಖಲೆ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_ದಾಖಲೆ(event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_ಬಳಸದ(backend);
    GGML_ಬಳಸದ(event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_ಗಣ_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ,
        size_t ಸ್ಥಾನಾಂತರ, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    ggml_backend_tensor_ಗಣ_2d(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ);
    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ,
        size_t ಸ್ಥಾನಾಂತರ, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    ggml_backend_tensor_get_2d(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ);
    GGML_ಬಳಸದ(backend);
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_ಗಣ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ಖಾಲಿ,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_rpc_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ggml_backend_rpc_event_ದಾಖಲೆ,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ ಖಾಲಿ,
};

ggml_backend_ಬಫರ್_type_t ggml_backend_rpc_ಬಫರ್_type(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * endpoint, uint32_t device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    ಸ್ಥಿರ std::unordered_map<std::string, ggml_backend_ಬಫರ್_type_t> buft_map;
    ಸ್ವಯಂಚಾಲಿತ it = buft_map.find(buft_name);
    ಆದರೆ (it != buft_map.end()) {
        ಹಿಂತಿರುಗಿ it->second;
    }
    ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_ಗಾತ್ರ(dispatcher, device);
    ggml_backend_rpc_ಬಫರ್_type_context * buft_ctx = ಹೊಸ ggml_backend_rpc_ಬಫರ್_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    ಸ್ವಯಂಚಾಲಿತ reg = ggml_backend_rpc_ಕೂಡಿಸು_server(endpoint);
    ggml_backend_ಬಫರ್_type_t buft = ಹೊಸ ggml_backend_ಬಫರ್_type {
        /* .iface   = */ ggml_backend_rpc_ಬಫರ್_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_ಗಾತ್ರ(alignment).c_str(), rpc_log_format_ಗಾತ್ರ(max_size).c_str());
    ಹಿಂತಿರುಗಿ buft;
}

ggml_backend_t ggml_backend_rpc_init(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = ಹೊಸ ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    ಸ್ವಯಂಚಾಲಿತ reg = ggml_backend_rpc_ಕೂಡಿಸು_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = ಹೊಸ ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    ಹಿಂತಿರುಗಿ backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_rpc(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

ಶೂನ್ಯ ggml_backend_rpc_get_device_ಸ್ಮರಣೆ(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * endpoint, uint32_t device, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(endpoint);
    ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_req>();
    request->device = device;
    rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_ಸ್ಮರಣೆ, request, ಗಾತ್ರ(*request), &response, ಗಾತ್ರ(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_ಗಾತ್ರ(response.free_mem).c_str(), rpc_log_format_ಗಾತ್ರ(response.ಒಟ್ಟು_mem).c_str());
    *ಬಿಡುಗಡೆ = response.free_mem;
    *ಒಟ್ಟು = response.ಒಟ್ಟು_mem;
}

// RPC server-side implementation

ವರ್ಗ rpc_server {
ಸಾರ್ವಜನಿಕ:
    rpc_server(std::vector<ggml_backend_t> all_backends, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_ಸ್ಥಿತಿಗಳು.resize(backends.size());
    }
    ~rpc_server();

    ಶೂನ್ಯ hello(rpc_ಸಂದೇಶ_hello_rsp & response);
    ತಾರ್ಕಿಕ alloc_ಬಫರ್(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_alloc_ಬಫರ್_req & request, rpc_ಸಂದೇಶ_alloc_ಬಫರ್_rsp & response);
    ತಾರ್ಕಿಕ get_alignment(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_alignment_req & request, rpc_ಸಂದೇಶ_get_alignment_rsp & response);
    ತಾರ್ಕಿಕ get_max_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_req & request, rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_rsp & response);
    ತಾರ್ಕಿಕ ಬಫರ್_get_ತಳ(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_req & request, rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_rsp & response);
    ತಾರ್ಕಿಕ free_ಬಫರ್(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_free_ಬಫರ್_req & request);
    ತಾರ್ಕಿಕ ಬಫರ್_clear(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_ಬಫರ್_clear_req & request);
    ತಾರ್ಕಿಕ memset_tensor(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_memset_tensor_req & request);
    ತಾರ್ಕಿಕ ಗಣ_tensor(ಸ್ಥಿರಾಂಕ std::vector<uint8_t> & ಒಳಹರಿವು);
    ತಾರ್ಕಿಕ ಗಣ_tensor_2d(ಸ್ಥಿರಾಂಕ std::vector<uint8_t> & ಒಳಹರಿವು);
    ತಾರ್ಕಿಕ ಗಣ_tensor_hash(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_req & request, rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_rsp & response);
    ತಾರ್ಕಿಕ get_tensor(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_tensor_req & request, std::vector<uint8_t> & response);
    ತಾರ್ಕಿಕ get_tensor_2d(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    ತಾರ್ಕಿಕ copy_tensor(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_copy_tensor_req & request, rpc_ಸಂದೇಶ_copy_tensor_rsp & response);
    ತಾರ್ಕಿಕ ನಕ್ಷೆ_compute(ಸ್ಥಿರಾಂಕ std::vector<uint8_t> & ಒಳಹರಿವು);
    ತಾರ್ಕಿಕ ನಕ್ಷೆ_recompute(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_ನಕ್ಷೆ_recompute_req & request);
    ತಾರ್ಕಿಕ comm_init(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_comm_init_req & request, rpc_ಸಂದೇಶ_comm_init_rsp & response);
    ತಾರ್ಕಿಕ comm_allreduce(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_comm_allreduce_req & request);
    ತಾರ್ಕಿಕ comm_free(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_comm_free_req & request);
    ತಾರ್ಕಿಕ init_tensor(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_init_tensor_req & request);
    ತಾರ್ಕಿಕ get_alloc_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_req & request, rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_rsp & response);
    ತಾರ್ಕಿಕ get_device_ಸ್ಮರಣೆ(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_req & request, rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_rsp & response);

    ರಚನೆ stored_ನಕ್ಷೆ {
        std::vector<uint8_t>   ಬಫರ್;
        ggml_cgraph          * graph;
    };

ಖಾಸಗಿ:
    ಶೂನ್ಯ sync_all_backends();
    ತಾರ್ಕಿಕ get_cached_file(uint64_t hash, std::vector<uint8_t> & ದತ್ತಾಂಶ);
    ggml_tensor * deserialize_tensor(ರಚನೆ ggml_context * ctx, ಸ್ಥಿರಾಂಕ rpc_tensor * tensor);
    ggml_tensor * create_ಗಂಟು(uint64_t id,
                              ರಚನೆ ggml_context * ctx,
                              ಸ್ಥಿರಾಂಕ std::unordered_map<uint64_t, ಸ್ಥಿರಾಂಕ rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, ರಚನೆ ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    ರಚನೆ comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_ಬಫರ್_ptr scratch;
        size_t                  scratch_ಗಾತ್ರ = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cache_dir;
    std::unordered_set<ggml_backend_ಬಫರ್_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_ನಕ್ಷೆ>> stored_graphs;
    std::vector<comm_state> comm_ಸ್ಥಿತಿಗಳು;
};

ಶೂನ್ಯ rpc_server::hello(rpc_ಸಂದೇಶ_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_ಆವೃತ್ತಿ;
    response.minor = RPC_PROTO_MINOR_ಆವೃತ್ತಿ;
    response.patch = RPC_PROTO_PATCH_ಆವೃತ್ತಿ;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

ತಾರ್ಕಿಕ rpc_server::get_alloc_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_req & request, rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_rsp & response) {
    uint32_t dev_id = request.device;
    ಆದರೆ (dev_id >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_backend_ಬಫರ್_type_t buft;
    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಆದರೆ (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (ಶೂನ್ಯ*)tensor->ಬಫರ್, tensor->ದತ್ತಾಂಶ);
    ಆದರೆ (tensor->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        //No buffer allocated.
        buft = ggml_backend_get_default_ಬಫರ್_type(backends[dev_id]);
    } ಇಲ್ಲದಿದ್ದರೆ {
        buft = tensor->ಬಫರ್->buft;
    }

    response.alloc_ಗಾತ್ರ = ggml_backend_buft_get_alloc_ಗಾತ್ರ(buft, tensor);

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::alloc_ಬಫರ್(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_alloc_ಬಫರ್_req & request, rpc_ಸಂದೇಶ_alloc_ಬಫರ್_rsp & response) {
    uint32_t dev_id = request.device;
    ಆದರೆ (dev_id >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_backend_ಬಫರ್_type_t buft = ggml_backend_get_default_ಬಫರ್_type(backends[dev_id]);
    ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_buft_alloc_ಬಫರ್(buft, request.size);
    response.remote_ptr = 0;
    response.remote_ಗಾತ್ರ = 0;
    ಆದರೆ (ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ) {
        response.remote_ptr = reinterpret_cast<uint64_t>(ಬಫರ್);
        response.remote_ಗಾತ್ರ = ಬಫರ್->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_ಗಾತ್ರ(request.size).c_str(), response.remote_ptr,
                rpc_log_format_ಗಾತ್ರ(response.remote_ಗಾತ್ರ).c_str());
        buffers.insert(ಬಫರ್);
    } ಇಲ್ಲದಿದ್ದರೆ {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_ಗಾತ್ರ(request.size).c_str());
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::get_alignment(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_alignment_req & request, rpc_ಸಂದೇಶ_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    ಆದರೆ (dev_id >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_backend_ಬಫರ್_type_t buft = ggml_backend_get_default_ಬಫರ್_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_ಗಾತ್ರ(alignment).c_str());
    response.alignment = alignment;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::get_max_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_req & request, rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_rsp & response) {
    uint32_t dev_id = request.device;
    ಆದರೆ (dev_id >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_backend_ಬಫರ್_type_t buft = ggml_backend_get_default_ಬಫರ್_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_ಗಾತ್ರ(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_ಗಾತ್ರ(max_size).c_str());
    response.max_size = max_size;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::ಬಫರ್_get_ತಳ(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_req & request, rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_ಬಫರ್_t ಬಫರ್ = reinterpret_cast<ggml_backend_ಬಫರ್_t>(request.remote_ptr);
    ಆದರೆ (buffers.find(ಬಫರ್) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಶೂನ್ಯ * ತಳ = ggml_backend_ಬಫರ್_get_ತಳ(ಬಫರ್);
    response.ತಳ_ptr = reinterpret_cast<uint64_t>(ತಳ);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::free_ಬಫರ್(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_free_ಬಫರ್_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_ಬಫರ್_t ಬಫರ್ = reinterpret_cast<ggml_backend_ಬಫರ್_t>(request.remote_ptr);
    ಆದರೆ (buffers.find(ಬಫರ್) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sgs : stored_graphs) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & sg : sgs) {
            sg.second.graph = ಶೂನ್ಯಸೂಚಕ;
        }
    }
    ggml_backend_ಬಫರ್_free(ಬಫರ್);
    buffers.erase(ಬಫರ್);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::ಬಫರ್_clear(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_ಬಫರ್_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_ಬಫರ್_t ಬಫರ್ = reinterpret_cast<ggml_backend_ಬಫರ್_t>(request.remote_ptr);
    ಆದರೆ (buffers.find(ಬಫರ್) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_backend_ಬಫರ್_clear(ಬಫರ್, request.value);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::memset_tensor(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_memset_tensor_req & request) {
    sync_all_backends();
    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ || tensor->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint64_t tensor_ಗಾತ್ರ = ggml_nbytes(tensor);
    ಆದರೆ (request.ಸ್ಥಾನಾಂತರ > tensor_ಗಾತ್ರ || request.size > tensor_ಗಾತ್ರ - request.ಸ್ಥಾನಾಂತರ) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.ಸ್ಥಾನಾಂತರ, request.size, tensor_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint64_t ಬಫರ್_ಆರಂಭ = (uint64_t) ggml_backend_ಬಫರ್_get_ತಳ(tensor->ಬಫರ್);
    ಸ್ಥಿರಾಂಕ uint64_t ಬಫರ್_ಗಾತ್ರ = ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(tensor->ಬಫರ್);
    ಆದರೆ (request.tensor.ದತ್ತಾಂಶ < ಬಫರ್_ಆರಂಭ) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ uint64_t ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ = request.tensor.ದತ್ತಾಂಶ - ಬಫರ್_ಆರಂಭ;
    ಆದರೆ (ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ > ಬಫರ್_ಗಾತ್ರ ||
        request.ಸ್ಥಾನಾಂತರ > ಬಫರ್_ಗಾತ್ರ - ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ ||
        request.size > ಬಫರ್_ಗಾತ್ರ - ದತ್ತಾಂಶ_ಸ್ಥಾನಾಂತರ - request.ಸ್ಥಾನಾಂತರ) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (tensor->ಬಫರ್->iface.memset_tensor == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ *) tensor->ಬಫರ್, tensor->ದತ್ತಾಂಶ, request.ಸ್ಥಾನಾಂತರ,
            rpc_log_format_ಗಾತ್ರ(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.ಸ್ಥಾನಾಂತರ, request.size);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ggml_tensor * rpc_server::deserialize_tensor(ರಚನೆ ggml_context * ctx, ಸ್ಥಿರಾಂಕ rpc_tensor * tensor) {
    // Validate tensor type before using it
    ಆದರೆ (tensor->type >= GGML_TYPE_ಎಣಿಕೆ) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    ಆದರೆ (ggml_blck_ಗಾತ್ರ((ಎಣಿಕೆ ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ggml_tensor * ಫಲಿತಾಂಶ = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    ಆದರೆ (ಫಲಿತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ಫಲಿತಾಂಶ->nb[i] = tensor->nb[i];
    }
    ಫಲಿತಾಂಶ->ಬಫರ್ = reinterpret_cast<ggml_backend_ಬಫರ್_t>(tensor->ಬಫರ್);
    ಆದರೆ (ಫಲಿತಾಂಶ->ಬಫರ್ && buffers.find(ಫಲಿತಾಂಶ->ಬಫರ್) == buffers.end()) {
        ಫಲಿತಾಂಶ->ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    }

    ಆದರೆ (ಫಲಿತಾಂಶ->ಬಫರ್ && ggml_nelements(ಫಲಿತಾಂಶ) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_ಗಾತ್ರ = (uint64_t) ggml_nbytes(ಫಲಿತಾಂಶ);
        uint64_t ಬಫರ್_ಆರಂಭ = (uint64_t) ggml_backend_ಬಫರ್_get_ತಳ(ಫಲಿತಾಂಶ->ಬಫರ್);
        uint64_t ಬಫರ್_ಗಾತ್ರ = (uint64_t) ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(ಫಲಿತಾಂಶ->ಬಫರ್);
        ಆದರೆ (tensor->ದತ್ತಾಂಶ + tensor_ಗಾತ್ರ < tensor->ದತ್ತಾಂಶ ||
            tensor->ದತ್ತಾಂಶ < ಬಫರ್_ಆರಂಭ || tensor->ದತ್ತಾಂಶ + tensor_ಗಾತ್ರ > ಬಫರ್_ಆರಂಭ + ಬಫರ್_ಗಾತ್ರ) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(ಫಲಿತಾಂಶ->type),
                      ಫಲಿತಾಂಶ->ne[0], ಫಲಿತಾಂಶ->ne[1], ಫಲಿತಾಂಶ->ne[2], ಫಲಿತಾಂಶ->ne[3],
                      tensor->ದತ್ತಾಂಶ, tensor->ದತ್ತಾಂಶ + tensor_ಗಾತ್ರ, ಬಫರ್_ಆರಂಭ, ಬಫರ್_ಆರಂಭ + ಬಫರ್_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }

    ಫಲಿತಾಂಶ->op = (ggml_op) tensor->op;
    ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t); i++) {
        ಫಲಿತಾಂಶ->op_params[i] = tensor->op_params[i];
    }
    ಫಲಿತಾಂಶ->flags = tensor->flags;
    ಫಲಿತಾಂಶ->ದತ್ತಾಂಶ = reinterpret_cast<ಶೂನ್ಯ *>(tensor->ದತ್ತಾಂಶ);
    ggml_ಗಣ_name(ಫಲಿತಾಂಶ, tensor->name);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}


ತಾರ್ಕಿಕ rpc_server::ಗಣ_tensor(ಸ್ಥಿರಾಂಕ std::vector<uint8_t> & ಒಳಹರಿವು) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ size_t ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ = ಗಾತ್ರ(rpc_tensor) + ಗಾತ್ರ(cache_flag) + ಗಾತ್ರ(ಸ್ಥಾನಾಂತರ);
    ಆದರೆ (ಒಳಹರಿವು.size() < ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ rpc_tensor * ಒಳ_tensor = (ಸ್ಥಿರಾಂಕ rpc_tensor *)ಒಳಹರಿವು.ದತ್ತಾಂಶ();
    ಸ್ಮರಣೆ_ನಕಲು(&cache_flag, ಒಳಹರಿವು.ದತ್ತಾಂಶ() + ಗಾತ್ರ(rpc_tensor), ಗಾತ್ರ(cache_flag));
    ಸ್ಮರಣೆ_ನಕಲು(&ಸ್ಥಾನಾಂತರ,     ಒಳಹರಿವು.ದತ್ತಾಂಶ() + ಗಾತ್ರ(rpc_tensor) + ಗಾತ್ರ(cache_flag), ಗಾತ್ರ(ಸ್ಥಾನಾಂತರ));
    ಸ್ಥಿರಾಂಕ size_t size = ಒಳಹರಿವು.size() - ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ;

    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, ಒಳ_tensor);
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ || tensor->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->ಬಫರ್, tensor->ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ,
            rpc_log_format_ಗಾತ್ರ(size).c_str());

    // sanitize tensor->data
    {
        ಸ್ಥಿರಾಂಕ size_t p0 = (size_t) ggml_backend_ಬಫರ್_get_ತಳ(tensor->ಬಫರ್);
        ಸ್ಥಿರಾಂಕ size_t p1 = p0 + ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(tensor->ಬಫರ್);

        ಆದರೆ (ಒಳ_tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ < p0 || ಒಳ_tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ >= p1 || size > (p1 - ಒಳ_tensor->ದತ್ತಾಂಶ - ಸ್ಥಾನಾಂತರ)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, ಒಳ_tensor->ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, p0, p1);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ = ಒಳಹರಿವು.ದತ್ತಾಂಶ() + ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ;
    ಆದರೆ (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ಸ್ಥಿರಾಂಕ uint8_t*)ದತ್ತಾಂಶ, size);
        ಅಕ್ಷರ hash_str[17];
        snprintf(hash_str, ಗಾತ್ರ(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.ಲಿಖ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)ದತ್ತಾಂಶ, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_ಗಣ(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::ಗಣ_tensor_2d(ಸ್ಥಿರಾಂಕ std::vector<uint8_t> & ಒಳಹರಿವು) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    ಆದರೆ (ಒಳಹರಿವು.size() < ಗಾತ್ರ(rpc_tensor) + 4*ಗಾತ್ರ(uint64_t)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ rpc_tensor * ಒಳ_tensor = (ಸ್ಥಿರಾಂಕ rpc_tensor *)ಒಳಹರಿವು.ದತ್ತಾಂಶ();
    uint64_t ಶೀರ್ಷಿಕೆ[4];
    ಸ್ಮರಣೆ_ನಕಲು(ಶೀರ್ಷಿಕೆ, ಒಳಹರಿವು.ದತ್ತಾಂಶ() + ಗಾತ್ರ(rpc_tensor), ಗಾತ್ರ(ಶೀರ್ಷಿಕೆ));
    ಸ್ಥಿರಾಂಕ uint64_t ಸ್ಥಾನಾಂತರ   = ಶೀರ್ಷಿಕೆ[0];
    ಸ್ಥಿರಾಂಕ uint64_t size     = ಶೀರ್ಷಿಕೆ[1];
    ಸ್ಥಿರಾಂಕ uint64_t n_copies = ಶೀರ್ಷಿಕೆ[2];
    ಸ್ಥಿರಾಂಕ uint64_t stride   = ಶೀರ್ಷಿಕೆ[3];

    ಸ್ಥಿರಾಂಕ uint64_t ದತ್ತಾಂಶ_ಗಾತ್ರ = ಒಳಹರಿವು.size() - ಗಾತ್ರ(rpc_tensor) - 4*ಗಾತ್ರ(uint64_t);
    ಆದರೆ (n_copies == 0 || size == 0 || size > ದತ್ತಾಂಶ_ಗಾತ್ರ / n_copies || size * n_copies != ದತ್ತಾಂಶ_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, ಒಳ_tensor);
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ || tensor->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->ಬಫರ್, tensor->ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ,
            rpc_log_format_ಗಾತ್ರ(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        ಆದರೆ (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ uint64_t ಹರವು = (n_copies - 1)*stride + size;
        ಸ್ಥಿರಾಂಕ uint64_t p0 = (uint64_t) ggml_backend_ಬಫರ್_get_ತಳ(tensor->ಬಫರ್);
        ಸ್ಥಿರಾಂಕ uint64_t p1 = p0 + ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(tensor->ಬಫರ್);

        ಆದರೆ (ಒಳ_tensor->ದತ್ತಾಂಶ < p0 || ಒಳ_tensor->ದತ್ತಾಂಶ > p1 || ಸ್ಥಾನಾಂತರ > p1 - ಒಳ_tensor->ದತ್ತಾಂಶ || ಹರವು > p1 - ಒಳ_tensor->ದತ್ತಾಂಶ - ಸ್ಥಾನಾಂತರ) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, ಒಳ_tensor->ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, ಹರವು, p0, p1);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ಸ್ಥಾನಾಂತರ > ggml_nbytes(tensor) || ಹರವು > ggml_nbytes(tensor) - ಸ್ಥಾನಾಂತರ) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, ಸ್ಥಾನಾಂತರ, ಹರವು, ggml_nbytes(tensor));
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ = ಒಳಹರಿವು.ದತ್ತಾಂಶ() + ಗಾತ್ರ(rpc_tensor) + 4*ಗಾತ್ರ(uint64_t);
    ggml_backend_tensor_ಗಣ_2d(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride, size);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & ದತ್ತಾಂಶ) {
    ಆದರೆ (!cache_dir) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಅಕ್ಷರ hash_str[17];
    snprintf(hash_str, ಗಾತ್ರ(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    ಆದರೆ (!fs::exists(cache_file, ec)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    ದತ್ತಾಂಶ.resize(size);
    ifs.ಪಠ((ಅಕ್ಷರ *)ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), size);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::ಗಣ_tensor_hash(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_req & request, rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_rsp & response)
{
    sync_all_backends();
    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ || tensor->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    std::vector<uint8_t> cached_file;
    ಆದರೆ (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.ಸ್ಥಾನಾಂತರ, request.hash);
        response.ಫಲಿತಾಂಶ = 0;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->ಬಫರ್, tensor->ದತ್ತಾಂಶ, request.ಸ್ಥಾನಾಂತರ,
            rpc_log_format_ಗಾತ್ರ(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ಸ್ಥಿರಾಂಕ size_t p0 = (size_t) ggml_backend_ಬಫರ್_get_ತಳ(tensor->ಬಫರ್);
        ಸ್ಥಿರಾಂಕ size_t p1 = p0 + ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(tensor->ಬಫರ್);

        ಆದರೆ (request.tensor.ದತ್ತಾಂಶ + request.ಸ್ಥಾನಾಂತರ < p0
         || request.tensor.ದತ್ತಾಂಶ + request.ಸ್ಥಾನಾಂತರ >= p1
         || size > (p1 - request.tensor.ದತ್ತಾಂಶ - request.ಸ್ಥಾನಾಂತರ)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.ದತ್ತಾಂಶ, request.ಸ್ಥಾನಾಂತರ, size, request.hash, p0, p1);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ggml_backend_tensor_ಗಣ(tensor, cached_file.ದತ್ತಾಂಶ(), request.ಸ್ಥಾನಾಂತರ, size);
    response.ಫಲಿತಾಂಶ = 1;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::init_tensor(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_init_tensor_req & request) {
    sync_all_backends();
    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (ಶೂನ್ಯ*)tensor->ಬಫರ್, tensor->ದತ್ತಾಂಶ);
    // Call the backend's buffer_init_tensor function
    ggml_backend_ಬಫರ್_t ಬಫರ್ = tensor->ಬಫರ್;
    ಆದರೆ (ಬಫರ್ && ಬಫರ್->iface.init_tensor) {
        ಬಫರ್->iface.init_tensor(ಬಫರ್, tensor);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (!ಬಫರ್) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    ಆದರೆ (tensor->extra != ಶೂನ್ಯಸೂಚಕ) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::get_tensor(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ || tensor->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->ಬಫರ್, tensor->ದತ್ತಾಂಶ, request.ಸ್ಥಾನಾಂತರ,
             rpc_log_format_ಗಾತ್ರ(request.size).c_str());

    // sanitize tensor->data
    {
        ಸ್ಥಿರಾಂಕ size_t p0 = (size_t) ggml_backend_ಬಫರ್_get_ತಳ(tensor->ಬಫರ್);
        ಸ್ಥಿರಾಂಕ size_t p1 = p0 + ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(tensor->ಬಫರ್);

        ಆದರೆ (request.tensor.ದತ್ತಾಂಶ + request.ಸ್ಥಾನಾಂತರ < p0 ||
            request.tensor.ದತ್ತಾಂಶ + request.ಸ್ಥಾನಾಂತರ >= p1 ||
            request.size > (p1 - request.tensor.ದತ್ತಾಂಶ - request.ಸ್ಥಾನಾಂತರ)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.ದತ್ತಾಂಶ, request.ಸ್ಥಾನಾಂತರ, request.size, p0, p1);
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.ದತ್ತಾಂಶ(), request.ಸ್ಥಾನಾಂತರ, request.size);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::get_tensor_2d(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ಆದರೆ (tensor == ಶೂನ್ಯಸೂಚಕ || tensor->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ಶೂನ್ಯ*)tensor->ಬಫರ್, tensor->ದತ್ತಾಂಶ, request.ಸ್ಥಾನಾಂತರ,
            rpc_log_format_ಗಾತ್ರ(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        ಆದರೆ (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ uint64_t ಹರವು = (request.n_copies - 1)*request.stride + request.size;
        ಸ್ಥಿರಾಂಕ uint64_t p0 = (uint64_t) ggml_backend_ಬಫರ್_get_ತಳ(tensor->ಬಫರ್);
        ಸ್ಥಿರಾಂಕ uint64_t p1 = p0 + ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(tensor->ಬಫರ್);

        ಆದರೆ (request.tensor.ದತ್ತಾಂಶ < p0 || request.tensor.ದತ್ತಾಂಶ > p1 || request.ಸ್ಥಾನಾಂತರ > p1 - request.tensor.ದತ್ತಾಂಶ ||
                ಹರವು > p1 - request.tensor.ದತ್ತಾಂಶ - request.ಸ್ಥಾನಾಂತರ) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.ದತ್ತಾಂಶ, request.ಸ್ಥಾನಾಂತರ, ಹರವು, p0, p1);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (request.ಸ್ಥಾನಾಂತರ > ggml_nbytes(tensor) || ಹರವು > ggml_nbytes(tensor) - request.ಸ್ಥಾನಾಂತರ) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.ಸ್ಥಾನಾಂತರ, ಹರವು, ggml_nbytes(tensor));
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.ದತ್ತಾಂಶ(), request.ಸ್ಥಾನಾಂತರ, request.size, request.n_copies, request.stride, request.size);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::copy_tensor(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_copy_tensor_req & request, rpc_ಸಂದೇಶ_copy_tensor_rsp & response) {
    sync_all_backends();
    ರಚನೆ ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    ಆದರೆ (src == ಶೂನ್ಯಸೂಚಕ || dst == ಶೂನ್ಯಸೂಚಕ || src->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ || dst->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    uint64_t src_ಗಾತ್ರ   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_ದತ್ತಾಂಶ   = (uint64_t) dst->ದತ್ತಾಂಶ;
    uint64_t dst_ತಳ   = (uint64_t) ggml_backend_ಬಫರ್_get_ತಳ(dst->ಬಫರ್);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(dst->ಬಫರ್);

    ಆದರೆ (dst_ದತ್ತಾಂಶ + src_ಗಾತ್ರ > dst_ತಳ + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_ದತ್ತಾಂಶ,
                         dst_ದತ್ತಾಂಶ + src_ಗಾತ್ರ,
                         dst_ತಳ,
                         dst_ತಳ + dst_buf_sz);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.ಫಲಿತಾಂಶ = ggml_backend_ಬಫರ್_copy_tensor(src, dst);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ggml_tensor * rpc_server::create_ಗಂಟು(uint64_t id,
                                      ರಚನೆ ggml_context * ctx,
                                      ಸ್ಥಿರಾಂಕ std::unordered_map<uint64_t, ಸ್ಥಿರಾಂಕ rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, ರಚನೆ ggml_tensor*> & tensor_map) {
    ಆದರೆ (tensor_map.find(id) != tensor_map.end()) {
        ಹಿಂತಿರುಗಿ tensor_map[id];
    }
    // Safely find the tensor pointer
    ಸ್ವಯಂಚಾಲಿತ it_ptr = tensor_ptrs.find(id);
    ಆದರೆ (it_ptr == tensor_ptrs.end()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಸ್ಥಿರಾಂಕ rpc_tensor * tensor = it_ptr->second;

    ರಚನೆ ggml_tensor * ಫಲಿತಾಂಶ = deserialize_tensor(ctx, tensor);
    ಆದರೆ (ಫಲಿತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (ಫಲಿತಾಂಶ->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ && ಫಲಿತಾಂಶ->ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    tensor_map[id] = ಫಲಿತಾಂಶ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        ಆದರೆ (tensor->src[i] == 0) {
            ಫಲಿತಾಂಶ->src[i] = ಶೂನ್ಯಸೂಚಕ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಫಲಿತಾಂಶ->src[i] = create_ಗಂಟು(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            ಆದರೆ (ಫಲಿತಾಂಶ->src[i] == ಶೂನ್ಯಸೂಚಕ) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
        }
    }

    // Handle view_src similarly
    ಆದರೆ (tensor->view_src == 0) {
        ಫಲಿತಾಂಶ->view_src = ಶೂನ್ಯಸೂಚಕ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ->view_src = create_ಗಂಟು(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        ಆದರೆ (ಫಲಿತಾಂಶ->view_src == ಶೂನ್ಯಸೂಚಕ) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }
    ಫಲಿತಾಂಶ->view_offs = tensor->view_offs;
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ rpc_server::ನಕ್ಷೆ_compute(ಸ್ಥಿರಾಂಕ std::vector<uint8_t> & ಒಳಹರಿವು) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    ಆದರೆ (ಒಳಹರಿವು.size() < 2*ಗಾತ್ರ(uint32_t) + ಗಾತ್ರ(uint64_t)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ uint8_t * src = ಒಳಹರಿವು.ದತ್ತಾಂಶ();
    uint32_t device;
    ಸ್ಮರಣೆ_ನಕಲು(&device, src, ಗಾತ್ರ(device));
    src += ಗಾತ್ರ(device);
    ಆದರೆ (device >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint64_t uid;
    ಸ್ಮರಣೆ_ನಕಲು(&uid, src, ಗಾತ್ರ(uid));
    src += ಗಾತ್ರ(uid);
    uint32_t n_nodes;
    ಸ್ಮರಣೆ_ನಕಲು(&n_nodes, src, ಗಾತ್ರ(n_nodes));
    src += ಗಾತ್ರ(n_nodes);
    ಆದರೆ (ಒಳಹರಿವು.size() < 2*ಗಾತ್ರ(uint32_t) + ಗಾತ್ರ(uint64_t) + n_nodes*ಗಾತ್ರ(uint64_t) + ಗಾತ್ರ(uint32_t)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ uint64_t * nodes = (ಸ್ಥಿರಾಂಕ uint64_t *)src;
    src += n_nodes*ಗಾತ್ರ(uint64_t);
    uint32_t n_tensors;
    ಸ್ಮರಣೆ_ನಕಲು(&n_tensors, src, ಗಾತ್ರ(n_tensors));
    src += ಗಾತ್ರ(n_tensors);
    ಆದರೆ (ಒಳಹರಿವು.size() < 2*ಗಾತ್ರ(uint32_t) + ಗಾತ್ರ(uint64_t) + n_nodes*ಗಾತ್ರ(uint64_t) + ಗಾತ್ರ(uint32_t) + n_tensors*ಗಾತ್ರ(rpc_tensor)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ rpc_tensor * tensors = (ಸ್ಥಿರಾಂಕ rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    ಆದರೆ (uid != 0 && stored_graphs[device].size() >= ನಕ್ಷೆ_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_ನಕ್ಷೆ sg_tmp;
    stored_ನಕ್ಷೆ & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_ಗಾತ್ರ = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_ನಕ್ಷೆ_overhead_custom(n_nodes, ಸುಳ್ಳು);
    ಆದರೆ (sg.ಬಫರ್.size() < buf_ಗಾತ್ರ) {
        sg.ಬಫರ್.resize(buf_ಗಾತ್ರ);
    }
    ರಚನೆ ggml_init_params params = {
        /*.mem_size   =*/ buf_ಗಾತ್ರ,
        /*.mem_buffer =*/ sg.ಬಫರ್.ದತ್ತಾಂಶ(),
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ರಚನೆ ggml_cgraph * graph = ggml_new_ನಕ್ಷೆ_custom(ctx, n_nodes, ಸುಳ್ಳು);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ಸ್ಥಿರಾಂಕ rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    ಗಾಗಿ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ t_deserialize = std::chrono::steady_clock::now();
    ಗಾಗಿ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        ಸ್ಮರಣೆ_ನಕಲು(&id, &nodes[i], ಗಾತ್ರ(id));
        graph->nodes[i] = create_ಗಂಟು(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        ಆದರೆ (graph->nodes[i] == ಶೂನ್ಯಸೂಚಕ && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (graph->nodes[i] != ಶೂನ್ಯಸೂಚಕ) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ಸ್ಥಿರಾಂಕ size_t hash_pos = ggml_hash_ಸೇರಿಸು(&graph->ಭೇಟಿಯಾದ_hash_ಗಣ, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_ಎಣಿಕೆ;
        }
    }
    ಸ್ಥಿರಾಂಕ std::chrono::duration<ದ್ವಿಗುಣ, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = ggml_backend_ನಕ್ಷೆ_compute_async(backends[device], graph);
    GGML_ASSERT(ಸ್ಥಿತಿ_2 == GGML_ಸ್ಥಿತಿ_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::ನಕ್ಷೆ_recompute(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_ನಕ್ಷೆ_recompute_req & request) {
    uint32_t device = request.device;
    ಆದರೆ (device >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ವಯಂಚಾಲಿತ it = stored_graphs[device].find(request.uid);
    ಆದರೆ (it == stored_graphs[device].end() || it->second.graph == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = ggml_backend_ನಕ್ಷೆ_compute_async(backends[device], graph);
    GGML_ASSERT(ಸ್ಥಿತಿ_2 == GGML_ಸ್ಥಿತಿ_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
ಶೂನ್ಯ rpc_server::sync_all_backends() {
    ಗಾಗಿ (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
ತಾರ್ಕಿಕ rpc_server::comm_init(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_comm_init_req & request, rpc_ಸಂದೇಶ_comm_init_rsp & response) {
    response.ok = 0;
    ಆದರೆ (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    comm_state & state = comm_ಸ್ಥಿತಿಗಳು[request.device];
    ಆದರೆ (state.peer != ಶೂನ್ಯಸೂಚಕ) {
        response.ok = 1;
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    uint8_t local_caps[RPC_CONN_CAPS_ಗಾತ್ರ] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_ಗಾತ್ರ] = {};
    ಆದರೆ (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.ಪೋರ್ಟ್);
        ಆದರೆ (srv == ಶೂನ್ಯಸೂಚಕ) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.ಪೋರ್ಟ್);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        state.peer = srv->accept();
        ಆದರೆ (state.peer == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (!state.peer->recv_ದತ್ತಾಂಶ(remote_caps, ಗಾತ್ರ(remote_caps))) {
            state.peer = ಶೂನ್ಯಸೂಚಕ;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        state.peer->get_caps(local_caps);
        ಆದರೆ (!state.peer->send_ದತ್ತಾಂಶ(local_caps, ಗಾತ್ರ(local_caps))) {
            state.peer = ಶೂನ್ಯಸೂಚಕ;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        state.peer->update_caps(remote_caps);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ std::string host(request.host, strnlen(request.host, ಗಾತ್ರ(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 100 && state.peer == ಶೂನ್ಯಸೂಚಕ; i++) {
            state.peer = socket_t::connect(host.c_str(), request.ಪೋರ್ಟ್);
            ಆದರೆ (state.peer == ಶೂನ್ಯಸೂಚಕ) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        ಆದರೆ (state.peer == ಶೂನ್ಯಸೂಚಕ) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.ಪೋರ್ಟ್);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        state.peer->get_caps(local_caps);
        ಆದರೆ (!state.peer->send_ದತ್ತಾಂಶ(local_caps, ಗಾತ್ರ(local_caps)) ||
            !state.peer->recv_ದತ್ತಾಂಶ(remote_caps, ಗಾತ್ರ(remote_caps))) {
            state.peer = ಶೂನ್ಯಸೂಚಕ;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::comm_allreduce(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_comm_allreduce_req & request) {
    ಆದರೆ (request.device >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    comm_state & state = comm_ಸ್ಥಿತಿಗಳು[request.device];
    ಆದರೆ (state.peer == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_ಗಾತ್ರ = 16*ggml_tensor_overhead() + 2*ggml_ನಕ್ಷೆ_overhead_custom(8, ಸುಳ್ಳು);
    ರಚನೆ ggml_init_params params = {
        /*.mem_size   =*/ ctx_ಗಾತ್ರ,
        /*.mem_buffer =*/ ಖಾಲಿ,
        /*.no_alloc   =*/ ನಿಜ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != ಶೂನ್ಯಸೂಚಕ);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    ಆದರೆ (t_dst == ಶೂನ್ಯಸೂಚಕ || t_dst->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ size_t  nbytes = ggml_nbytes(t_dst);
    ಸ್ಥಿರಾಂಕ int64_t ne     = ggml_nelements(t_dst);
    ಆದರೆ (nbytes == 0) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ಸ್ಥಿರಾಂಕ size_t wire_ಬೈಟುಗಳು = wire_bf16 ? (size_t) ne*2 : nbytes;
    ಸ್ಥಿರಾಂಕ size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    ಆದರೆ (state.scratch_ಗಾತ್ರ < need) {
        state.scratch.reset(ggml_backend_alloc_ಬಫರ್(backend, need));
        state.scratch_ಗಾತ್ರ = need;
    }
    ಅಕ್ಷರ * scratch_ತಳ = (ಅಕ್ಷರ *) ggml_backend_ಬಫರ್_get_ತಳ(state.scratch.get());
    state.send_buf.resize(wire_ಬೈಟುಗಳು);
    state.recv_buf.resize(wire_ಬೈಟುಗಳು);

    ಸ್ವಯಂಚಾಲಿತ new_scratch_tensor = [&](ggml_type type, size_t ಸ್ಥಾನಾಂತರ) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->ಬಫರ್ = state.scratch.get();
        t->ದತ್ತಾಂಶ   = scratch_ತಳ + ಸ್ಥಾನಾಂತರ;
        ಹಿಂತಿರುಗಿ t;
    };
    ಸ್ವಯಂಚಾಲಿತ new_cpy_ಗಂಟು = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->ಬಫರ್ = dst->ಬಫರ್;
        t->ದತ್ತಾಂಶ   = dst->ದತ್ತಾಂಶ;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ಹಿಂತಿರುಗಿ t;
    };
    ಸ್ವಯಂಚಾಲಿತ compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_ನಕ್ಷೆ_custom(ctx, 2, ಸುಳ್ಳು);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != ಶೂನ್ಯಸೂಚಕ ? 2 : 1;
        ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = ggml_backend_ನಕ್ಷೆ_compute_async(backend, graph);
        GGML_ASSERT(ಸ್ಥಿತಿ_2 == GGML_ಸ್ಥಿತಿ_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = ಶೂನ್ಯಸೂಚಕ;
    ggml_tensor * t_wire_recv = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_ಗಂಟು(t_dst, t_wire_send), ಶೂನ್ಯಸೂಚಕ);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.ದತ್ತಾಂಶ(), 0, wire_ಬೈಟುಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_backend_tensor_get(t_dst, state.send_buf.ದತ್ತಾಂಶ(), 0, wire_ಬೈಟುಗಳು);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    ಆದರೆ (state.rank == 0) {
        ಆದರೆ (!state.peer->send_ದತ್ತಾಂಶ(state.send_buf.ದತ್ತಾಂಶ(), wire_ಬೈಟುಗಳು) || !state.peer->flush() ||
            !state.peer->recv_ದತ್ತಾಂಶ(state.recv_buf.ದತ್ತಾಂಶ(), wire_ಬೈಟುಗಳು)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (!state.peer->recv_ದತ್ತಾಂಶ(state.recv_buf.ದತ್ತಾಂಶ(), wire_ಬೈಟುಗಳು) ||
            !state.peer->send_ದತ್ತಾಂಶ(state.send_buf.ದತ್ತಾಂಶ(), wire_ಬೈಟುಗಳು) || !state.peer->flush()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (wire_bf16) {
        ggml_backend_tensor_ಗಣ(t_wire_recv, state.recv_buf.ದತ್ತಾಂಶ(), 0, wire_ಬೈಟುಗಳು);
        t_cast = new_cpy_ಗಂಟು(t_wire_recv, t_peer);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_backend_tensor_ಗಣ(t_peer, state.recv_buf.ದತ್ತಾಂಶ(), 0, wire_ಬೈಟುಗಳು);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ಕೂಡಿಸು;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->ಬಫರ್ = t_dst->ಬಫರ್;
    t_red->ದತ್ತಾಂಶ   = t_dst->ದತ್ತಾಂಶ;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    ಆದರೆ (t_cast != ಶೂನ್ಯಸೂಚಕ) {
        compute_nodes(t_cast, t_red);
    } ಇಲ್ಲದಿದ್ದರೆ {
        compute_nodes(t_red, ಶೂನ್ಯಸೂಚಕ);
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::comm_free(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_comm_free_req & request) {
    ಆದರೆ (request.device >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    comm_ಸ್ಥಿತಿಗಳು[request.device] = comm_state();
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ rpc_server::get_device_ಸ್ಮರಣೆ(ಸ್ಥಿರಾಂಕ rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_req & request, rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_rsp & response) {
    uint32_t dev_id = request.device;
    ಆದರೆ (dev_id >= backends.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    size_t ಬಿಡುಗಡೆ, ಒಟ್ಟು;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_ಸ್ಮರಣೆ(dev, &ಬಿಡುಗಡೆ, &ಒಟ್ಟು);
    response.free_mem = ಬಿಡುಗಡೆ;
    response.ಒಟ್ಟು_mem = ಒಟ್ಟು;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_ಗಾತ್ರ(response.free_mem).c_str(), rpc_log_format_ಗಾತ್ರ(response.ಒಟ್ಟು_mem).c_str());
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

rpc_server::~rpc_server() {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ ಬಫರ್ : buffers) {
        ggml_backend_ಬಫರ್_free(ಬಫರ್);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rpc_serve_client(ಸ್ಥಿರಾಂಕ std::vector<ggml_backend_t> & backends, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    ಆದರೆ (!sock->recv_ದತ್ತಾಂಶ(&cmd, 1)) {
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        ಹಿಂತಿರುಗಿ;
    }

    // Read input_size and validate protocol version
    uint64_t hello_ಒಳಹರಿವು_ಗಾತ್ರ;
    ಆದರೆ (!sock->recv_ದತ್ತಾಂಶ(&hello_ಒಳಹರಿವು_ಗಾತ್ರ, ಗಾತ್ರ(hello_ಒಳಹರಿವು_ಗಾತ್ರ))) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (hello_ಒಳಹರಿವು_ಗಾತ್ರ != ಗಾತ್ರ(rpc_ಸಂದೇಶ_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_ಒಳಹರಿವು_ಗಾತ್ರ, ಗಾತ್ರ(rpc_ಸಂದೇಶ_hello_req), RPC_PROTO_MAJOR_ಆವೃತ್ತಿ);
        ಹಿಂತಿರುಗಿ;
    }

    rpc_ಸಂದೇಶ_hello_req req = {};
    ಆದರೆ (!sock->recv_ದತ್ತಾಂಶ(&req, ಗಾತ್ರ(req))) {
        ಹಿಂತಿರುಗಿ;
    }

    rpc_ಸಂದೇಶ_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    ಆದರೆ (!send_ಸಂದೇಶ(sock, &rsp, ಗಾತ್ರ(rsp))) {
        ಹಿಂತಿರುಗಿ;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    ಆಗ (ನಿಜ) {
        ಆದರೆ (!sock->recv_ದತ್ತಾಂಶ(&cmd, 1)) {
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (cmd >= RPC_CMD_ಎಣಿಕೆ) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            ನಿಲ್ಲಿಸು;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((ಎಣಿಕೆ rpc_cmd)cmd));
        ಆಯ್ಕೆ (cmd) {
            ಸ್ಥಿತಿ RPC_CMD_HELLO: {
                // HELLO command is handled above
                ಹಿಂತಿರುಗಿ;
            }
            ಸ್ಥಿತಿ RPC_CMD_DEVICE_ಎಣಿಕೆ: {
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, ಶೂನ್ಯಸೂಚಕ, 0)) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_device_ಎಣಿಕೆ_rsp response;
                response.device_ಎಣಿಕೆ = backends.size();
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_ALLOC_ಬಫರ್: {
                rpc_ಸಂದೇಶ_alloc_ಬಫರ್_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_alloc_ಬಫರ್_rsp response;
                ಆದರೆ (!server.alloc_ಬಫರ್(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_GET_ALLOC_ಗಾತ್ರ: {
                rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_get_alloc_ಗಾತ್ರ_rsp response;
                ಆದರೆ (!server.get_alloc_ಗಾತ್ರ(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_GET_ALIGNMENT: {
                rpc_ಸಂದೇಶ_get_alignment_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_get_alignment_rsp response;
                ಆದರೆ (!server.get_alignment(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_GET_MAX_ಗಾತ್ರ: {
                rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_get_max_ಗಾತ್ರ_rsp response;
                ಆದರೆ (!server.get_max_ಗಾತ್ರ(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_ಬಫರ್_GET_ತಳ: {
                rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_ಬಫರ್_get_ತಳ_rsp response;
                ಆದರೆ (!server.ಬಫರ್_get_ತಳ(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_FREE_ಬಫರ್: {
                rpc_ಸಂದೇಶ_free_ಬಫರ್_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.free_ಬಫರ್(request)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_ಬಫರ್_CLEAR: {
                rpc_ಸಂದೇಶ_ಬಫರ್_clear_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.ಬಫರ್_clear(request)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_MEMSET_TENSOR: {
                rpc_ಸಂದೇಶ_memset_tensor_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.memset_tensor(request)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_ಗಣ_TENSOR: {
                std::vector<uint8_t> ಒಳಹರಿವು;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, ಒಳಹರಿವು)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.ಗಣ_tensor(ಒಳಹರಿವು)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_ಗಣ_TENSOR_2_D: {
                std::vector<uint8_t> ಒಳಹರಿವು;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, ಒಳಹರಿವು)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.ಗಣ_tensor_2d(ಒಳಹರಿವು)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_GET_TENSOR_2D: {
                rpc_ಸಂದೇಶ_get_tensor_2d_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                std::vector<uint8_t> response;
                ಆದರೆ (!server.get_tensor_2d(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, response.ದತ್ತಾಂಶ(), response.size())) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_ಗಣ_TENSOR_HASH: {
                rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_ಗಣ_tensor_hash_rsp response;
                ಆದರೆ (!server.ಗಣ_tensor_hash(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_INIT_TENSOR: {
                rpc_ಸಂದೇಶ_init_tensor_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request,ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.init_tensor(request)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_GET_TENSOR: {
                rpc_ಸಂದೇಶ_get_tensor_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                std::vector<uint8_t> response;
                ಆದರೆ (!server.get_tensor(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, response.ದತ್ತಾಂಶ(), response.size())) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_COPY_TENSOR: {
                rpc_ಸಂದೇಶ_copy_tensor_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_copy_tensor_rsp response;
                ಆದರೆ (!server.copy_tensor(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_ನಕ್ಷೆ_COMPUTE: {
                std::vector<uint8_t> ಒಳಹರಿವು;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, ಒಳಹರಿವು)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.ನಕ್ಷೆ_compute(ಒಳಹರಿವು)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_ನಕ್ಷೆ_RECOMPUTE: {
                rpc_ಸಂದೇಶ_ನಕ್ಷೆ_recompute_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.ನಕ್ಷೆ_recompute(request)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_COMM_INIT: {
                rpc_ಸಂದೇಶ_comm_init_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_comm_init_rsp response;
                ಆದರೆ (!server.comm_init(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_COMM_ALLREDUCE: {
                rpc_ಸಂದೇಶ_comm_allreduce_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.comm_allreduce(request)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_COMM_FREE: {
                rpc_ಸಂದೇಶ_comm_free_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!server.comm_free(request)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ RPC_CMD_GET_DEVICE_ಸ್ಮರಣೆ: {
                rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_req request;
                ಆದರೆ (!recv_ಸಂದೇಶ(sock, &request, ಗಾತ್ರ(request))) {
                    ಹಿಂತಿರುಗಿ;
                }
                rpc_ಸಂದೇಶ_get_device_ಸ್ಮರಣೆ_rsp response;
                ಆದರೆ (!server.get_device_ಸ್ಮರಣೆ(request, response)) {
                    ಹಿಂತಿರುಗಿ;
                }
                ಆದರೆ (!send_ಸಂದೇಶ(sock, &response, ಗಾತ್ರ(response))) {
                    ಹಿಂತಿರುಗಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಪೂರ್ವನಿಯೋಜಿತ: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((ಎಣಿಕೆ rpc_cmd)cmd));
                ಹಿಂತಿರುಗಿ;
            }
        }
    }
}

ಶೂನ್ಯ ggml_backend_rpc_ಆರಂಭ_server(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * endpoint, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * cache_dir,
                                   size_t n_ಎಳೆಗಳು, size_t n_devices, ggml_backend_dev_t * devices) {
    ಆದರೆ (n_devices == 0 || devices == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        ಹಿಂತಿರುಗಿ;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_ಆವೃತ್ತಿ,
        RPC_PROTO_MINOR_ಆವೃತ್ತಿ,
        RPC_PROTO_PATCH_ಆವೃತ್ತಿ);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    ಗಾಗಿ (size_t i = 0; i < n_devices; i++) {
        ಸ್ವಯಂಚಾಲಿತ dev = devices[i];
        size_t ಬಿಡುಗಡೆ, ಒಟ್ಟು;
        ggml_backend_dev_ಸ್ಮರಣೆ(dev, &ಬಿಡುಗಡೆ, &ಒಟ್ಟು);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 ಒಟ್ಟು / 1024 / 1024, ಬಿಡುಗಡೆ / 1024 / 1024);
        ಸ್ವಯಂಚಾಲಿತ backend = ggml_backend_dev_init(dev, ಶೂನ್ಯಸೂಚಕ);
        ಆದರೆ (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            ಹಿಂತಿರುಗಿ;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : ಶೂನ್ಯಸೂಚಕ;
        ಆದರೆ (reg) {
            ಸ್ವಯಂಚಾಲಿತ ggml_backend_ಗಣ_n_ಎಳೆಗಳು_fn = (ggml_backend_ಗಣ_n_ಎಳೆಗಳು_t) ggml_backend_reg_get_proc_ವಿಳಾಸ(reg, "ggml_backend_set_n_threads");
            ಆದರೆ (ggml_backend_ಗಣ_n_ಎಳೆಗಳು_fn) {
                ggml_backend_ಗಣ_n_ಎಳೆಗಳು_fn(backend, n_ಎಳೆಗಳು);
            }
        }
    }

    std::string host;
    ಪೂರ್ಣಾಂಕ ಪೋರ್ಟ್;
    ಆದರೆ (!parse_endpoint(endpoint, host, ಪೋರ್ಟ್)) {
        ಹಿಂತಿರುಗಿ;
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#ಇಲ್ಲದಿದ್ದರೆ
    LOG_INFO("  transport      : TCP\n");
#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA
    ಆದರೆ (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ವಯಂಚಾಲಿತ server_socket = socket_t::create_server(host.c_str(), ಪೋರ್ಟ್);
    ಆದರೆ (server_socket == ಶೂನ್ಯಸೂಚಕ) {
        LOG_ERROR("Failed to create server socket\n");
        ಹಿಂತಿರುಗಿ;
    }
    ಆಗ (ನಿಜ) {
        ಸ್ವಯಂಚಾಲಿತ client_socket = server_socket->accept();
        ಆದರೆ (client_socket == ಶೂನ್ಯಸೂಚಕ) {
            LOG_ERROR("Failed to accept client connection\n");
            ಹಿಂತಿರುಗಿ;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ backend : backends) {
        ggml_backend_free(backend);
    }
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ಹಿಂತಿರುಗಿ ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_ಸ್ಮರಣೆ(ctx->endpoint.c_str(), ctx->device, ಬಿಡುಗಡೆ, ಒಟ್ಟು);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->caps = {
        /* .async                 = */ ನಿಜ,
        /* .host_buffer           = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr  = */ ಸುಳ್ಳು,
        /* .events                = */ ನಿಜ,
        /* .mmap_support          = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ಹಿಂತಿರುಗಿ ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_ಬಳಸದ(params);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_rpc_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ಹಿಂತಿರುಗಿ ggml_backend_rpc_ಬಫರ್_type(ctx->endpoint.c_str(), ctx->device);

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    GGML_ಬಳಸದ(dev);
    GGML_ಬಳಸದ(op);
    //TODO: call the remote backend and cache the results
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಆದರೆ (!buft || buft->iface.get_name != ggml_backend_rpc_ಬಫರ್_type_name) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_backend_rpc_ಬಫರ್_type_context * buft_ctx = (ggml_backend_rpc_ಬಫರ್_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(ctx->endpoint);
    ಹಿಂತಿರುಗಿ dispatcher->event_new(dev);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ಖಾಲಿ,
    /* .buffer_from_host_ptr = */ ಖಾಲಿ,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ ಖಾಲಿ,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

ರಚನೆ ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ಹಿಂತಿರುಗಿ ctx ? ctx->name.c_str() : "RPC";
}

ಸ್ಥಿರ size_t ggml_backend_rpc_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ಹಿಂತಿರುಗಿ ctx ? ctx->devices.size() : 0;
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ಆದರೆ (ctx == ಶೂನ್ಯಸೂಚಕ) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(ಸೂಚ್ಯಂಕ < ctx->devices.size());
        ಹಿಂತಿರುಗಿ ctx->devices[ಸೂಚ್ಯಂಕ];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
ರಚನೆ ggml_backend_rpc_comm_context {
    ರಚನೆ rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_rpc_comm_free(ಶೂನ್ಯ * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ಆದರೆ (comm_ctx == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & rank : comm_ctx->ranks) {
        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, ಗಾತ್ರ(*request));
        rank.dispatcher->busy_spin_release();
    }
    ಅಳಿಸು comm_ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ಆದರೆ (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ಆದರೆ (!ggml_backend_is_rpc(backends[i])) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & rank : ranks) {
            ಆದರೆ (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    ಪೂರ್ಣಾಂಕ port0;
    ಆದರೆ (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಸ್ಥಿರಾಂಕ uint32_t comm_ಪೋರ್ಟ್ = (uint32_t) port0 + 1000;
    ಆದರೆ (host0.size() >= 64) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_ಸಂದೇಶ_comm_init_rsp> responses(n_backends);
    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->ಪೋರ್ಟ್   = comm_ಪೋರ್ಟ್;
        ಆದರೆ (i > 0) {
            ಸ್ಮರಣೆ_ನಕಲು(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, ಗಾತ್ರ(*request), &responses[i], ಗಾತ್ರ(responses[i]));
    }
    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    ತಾರ್ಕಿಕ ok = ನಿಜ;
    ಗಾಗಿ (size_t i = 0; i < n_backends; i++) {
        ಆದರೆ (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = ಸುಳ್ಳು;
        }
    }
    ಆದರೆ (!ok) {
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_backend_rpc_comm_context{std::move(ranks)};
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_rpc_comm_allreduce_tensor(ಶೂನ್ಯ * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ಆದರೆ (comm_ctx == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ size_t n_ranks = comm_ctx->ranks.size();
    ಸ್ಥಿರಾಂಕ int64_t ne = ggml_nelements(tensors[0]);
    ಆದರೆ (ne == 0) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಗಾಗಿ (size_t i = 0; i < n_ranks; i++) {
        ಆದರೆ (tensors[i] == ಶೂನ್ಯಸೂಚಕ || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ || !ggml_backend_ಬಫರ್_is_rpc(tensors[i]->ಬಫರ್)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        ಆದರೆ ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಗಾಗಿ (size_t i = 0; i < n_ranks; i++) {
        ಸ್ವಯಂಚಾಲಿತ request = std::make_shared<rpc_ಸಂದೇಶ_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, ಗಾತ್ರ(*request));
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_rpc_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಆದರೆ (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_rpc_ಕೂಡಿಸು_server;
    }
    ಆದರೆ (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_rpc_ಆರಂಭ_server;
    }
    ಆದರೆ (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_rpc_comm_init;
    }
    ಆದರೆ (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_rpc_comm_free;
    }
    ಆದರೆ (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    ಹಿಂತಿರುಗಿ ಖಾಲಿ;

    GGML_ಬಳಸದ(reg);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_ವಿಳಾಸ,
};

ggml_backend_reg_t ggml_backend_rpc_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ ಖಾಲಿ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_rpc_reg;
}

ಸ್ಥಿರ uint32_t ggml_backend_rpc_get_device_ಎಣಿಕೆ(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * endpoint) {
    ಸ್ವಯಂಚಾಲಿತ dispatcher = get_dispatcher(endpoint);
    rpc_ಸಂದೇಶ_device_ಎಣಿಕೆ_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_ಎಣಿಕೆ, ಶೂನ್ಯಸೂಚಕ, 0, &response, ಗಾತ್ರ(response));
    ಹಿಂತಿರುಗಿ response.device_ಎಣಿಕೆ;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_ಎಣಿಕೆ,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_ವಿಳಾಸ,
};

ggml_backend_reg_t ggml_backend_rpc_ಕೂಡಿಸು_server(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * endpoint) {
    ಸ್ಥಿರ std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    ಸ್ಥಿರ std::mutex mutex;
    ಸ್ಥಿರ uint32_t dev_id = 0;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);
    ಆದರೆ (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        ಹಿಂತಿರುಗಿ reg_map[endpoint];
    }
    uint32_t dev_ಎಣಿಕೆ = ggml_backend_rpc_get_device_ಎಣಿಕೆ(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_ಎಣಿಕೆ);
    ಆದರೆ (dev_ಎಣಿಕೆ == 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ggml_backend_rpc_reg_context * ctx = ಹೊಸ ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    ಗಾಗಿ (uint32_t ind = 0; ind < dev_ಎಣಿಕೆ; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = ಹೊಸ ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = ಹೊಸ ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = ಹೊಸ ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    ಹಿಂತಿರುಗಿ reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

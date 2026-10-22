#ଅନ୍ତର୍ଭାଵ "ggml-rpc.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpp.h"
#ଅନ୍ତର୍ଭାଵ "transport.h"
#ଅନ୍ତର୍ଭାଵ "log.h"

#ଅନ୍ତର୍ଭାଵ <array>
#ଅନ୍ତର୍ଭାଵ <chrono>
#ଅନ୍ତର୍ଭାଵ <cinttypes>
#ଅନ୍ତର୍ଭାଵ <optional>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <thread>
#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <queue>
#ଅନ୍ତର୍ଭାଵ <condition_variable>
#ଅନ୍ତର୍ଭାଵ <future>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <mutex>
#ଅନ୍ତର୍ଭାଵ <unordered_map>
#ଅନ୍ତର୍ଭାଵ <unordered_set>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <filesystem>
#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <atomic>

ନାମାକାଶ fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) ଯଦି (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(ଠେଲ, 1)
// ggml_tensor is serialized into rpc_tensor
ସଂରଚନା rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / ଆକାର(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    ଅକ୍ଷର name[GGML_MAX_NAME];

    int32_t use_ଗଣନା;
};

static_assert(ଆକାର(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
ଗଣନା rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_ଗଣନା,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_ଗଣନା,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * rpc_cmd_name(ଗଣନା rpc_cmd cmd) {
    ବାଛ (cmd) {
        ସ୍ଥିତି RPC_CMD_ALLOC_BUFFER:      ଫେରନ୍ତୁ "ALLOC_BUFFER";
        ସ୍ଥିତି RPC_CMD_GET_ALIGNMENT:     ଫେରନ୍ତୁ "GET_ALIGNMENT";
        ସ୍ଥିତି RPC_CMD_GET_MAX_SIZE:      ଫେରନ୍ତୁ "GET_MAX_SIZE";
        ସ୍ଥିତି RPC_CMD_BUFFER_GET_BASE:   ଫେରନ୍ତୁ "BUFFER_GET_BASE";
        ସ୍ଥିତି RPC_CMD_FREE_BUFFER:       ଫେରନ୍ତୁ "FREE_BUFFER";
        ସ୍ଥିତି RPC_CMD_BUFFER_CLEAR:      ଫେରନ୍ତୁ "BUFFER_CLEAR";
        ସ୍ଥିତି RPC_CMD_SET_TENSOR:        ଫେରନ୍ତୁ "SET_TENSOR";
        ସ୍ଥିତି RPC_CMD_SET_TENSOR_HASH:   ଫେରନ୍ତୁ "SET_TENSOR_HASH";
        ସ୍ଥିତି RPC_CMD_GET_TENSOR:        ଫେରନ୍ତୁ "GET_TENSOR";
        ସ୍ଥିତି RPC_CMD_COPY_TENSOR:       ଫେରନ୍ତୁ "COPY_TENSOR";
        ସ୍ଥିତି RPC_CMD_GRAPH_COMPUTE:     ଫେରନ୍ତୁ "GRAPH_COMPUTE";
        ସ୍ଥିତି RPC_CMD_GET_DEVICE_MEMORY: ଫେରନ୍ତୁ "GET_DEVICE_MEMORY";
        ସ୍ଥିତି RPC_CMD_INIT_TENSOR:       ଫେରନ୍ତୁ "INIT_TENSOR";
        ସ୍ଥିତି RPC_CMD_GET_ALLOC_SIZE:    ଫେରନ୍ତୁ "GET_ALLOC_SIZE";
        ସ୍ଥିତି RPC_CMD_HELLO:             ଫେରନ୍ତୁ "HELLO";
        ସ୍ଥିତି RPC_CMD_DEVICE_ଗଣନା:      ଫେରନ୍ତୁ "DEVICE_COUNT";
        ସ୍ଥିତି RPC_CMD_GRAPH_RECOMPUTE:   ଫେରନ୍ତୁ "GRAPH_RECOMPUTE";
        ସ୍ଥିତି RPC_CMD_MEMSET_TENSOR:     ଫେରନ୍ତୁ "MEMSET_TENSOR";
        ସ୍ଥିତି RPC_CMD_SET_TENSOR_2D:     ଫେରନ୍ତୁ "SET_TENSOR_2D";
        ସ୍ଥିତି RPC_CMD_GET_TENSOR_2D:     ଫେରନ୍ତୁ "GET_TENSOR_2D";
        ସ୍ଥିତି RPC_CMD_COMM_INIT:         ଫେରନ୍ତୁ "COMM_INIT";
        ସ୍ଥିତି RPC_CMD_COMM_ALLREDUCE:    ଫେରନ୍ତୁ "COMM_ALLREDUCE";
        ସ୍ଥିତି RPC_CMD_COMM_FREE:         ଫେରନ୍ତୁ "COMM_FREE";
        ସ୍ଥିତି RPC_CMD_NONE:              ଫେରନ୍ତୁ "NONE";
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:                        ଫେରନ୍ତୁ "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ଅଚଳ size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ଅଚଳ size_t GRAPH_CACHE_MAX = 1024;

ସଂରଚନା rpc_ସନ୍ଦେଶ_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_device_ଗଣନା_rsp {
    uint32_t device_ଗଣନା;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_alloc_size_rsp {
    uint64_t alloc_size;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_init_tensor_req {
    rpc_tensor tensor;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_alignment_req {
    uint32_t device;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_alignment_rsp {
    uint64_t alignment;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_max_size_req {
    uint32_t device;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_max_size_rsp {
    uint64_t max_size;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_buffer_get_base_req {
    uint64_t remote_ptr;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_buffer_get_base_rsp {
    uint64_t base_ptr;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_free_buffer_req {
    uint64_t remote_ptr;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_set_tensor_hash_rsp {
    uint8_t result;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_copy_tensor_rsp {
    uint8_t result;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_device_memory_req {
    uint32_t device;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    ଅକ୍ଷର     host[64];  // rank > 0: rank 0's host
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_comm_init_rsp {
    uint8_t ok;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

ସଂରଚନା rpc_ସନ୍ଦେଶ_comm_free_req {
    uint32_t device;
};

#pragma pack(କାଢ଼)

// RPC data structures

ସ୍ଥିର ggml_guid_t ggml_backend_rpc_guid() {
    ସ୍ଥିର ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    ଫେରନ୍ତୁ &guid;
}

ସଂରଚନା ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

ସଂରଚନା ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

ଶ୍ରେଣୀ rpc_dispatcher;
ସଂରଚନା ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

ସଂରଚନା ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    ଶୂନ୍ୟ                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
ସ୍ଥିର uint64_t fnv_hash(ଅଚଳ uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ଅଚଳ uint64_t fnv_ମୌଳିକ = 0x100000001b3ULL;

    ପାଇଁ (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_ମୌଳିକ;
    }
    ଫେରନ୍ତୁ hash;
}

ସ୍ଥିର ତାର୍କିକ send_ସନ୍ଦେଶ(socket_ptr sock, ଅଚଳ ଶୂନ୍ୟ * ସନ୍ଦେଶ, size_t ସନ୍ଦେଶ_size) {
    ଯଦି (!sock->send_data(&ସନ୍ଦେଶ_size, ଆକାର(ସନ୍ଦେଶ_size))) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!sock->send_data(ସନ୍ଦେଶ, ସନ୍ଦେଶ_size)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ sock->flush();
}

ସ୍ଥିର ତାର୍କିକ recv_ସନ୍ଦେଶ(socket_ptr sock, ଶୂନ୍ୟ * ସନ୍ଦେଶ, size_t ସନ୍ଦେଶ_size) {
    uint64_t size;
    ଯଦି (!sock->recv_data(&size, ଆକାର(size))) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (size != ସନ୍ଦେଶ_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ sock->recv_data(ସନ୍ଦେଶ, ସନ୍ଦେଶ_size);
}

ସ୍ଥିର ତାର୍କିକ recv_ସନ୍ଦେଶ(socket_ptr sock, std::vector<uint8_t> & ନିବେଶ) {
    uint64_t size;
    ଯଦି (!sock->recv_data(&size, ଆକାର(size))) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଚେଷ୍ଟା {
        ନିବେଶ.resize(size);
    } ଧର (ଅଚଳ std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ sock->recv_data(ନିବେଶ.data(), size);
}

ସ୍ଥିର ତାର୍କିକ parse_endpoint(ଅଚଳ std::string & endpoint, std::string & host, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & port) {
    size_t pos = endpoint.find(':');
    ଯଦି (pos == std::string::npos) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    host = endpoint.substr(0, pos);
    ଚେଷ୍ଟା {
        port = std::stoi(endpoint.substr(pos + 1));
    } ଧର (...) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
ସ୍ଥିର ତାର୍କିକ send_rpc_cmd(socket_ptr sock, ଗଣନା rpc_cmd cmd, ଅଚଳ ଶୂନ୍ୟ * ନିବେଶ, size_t ନିବେଶ_size) {
    uint8_t cmd_byte = cmd;
    ଯଦି (!sock->send_data(&cmd_byte, ଆକାର(cmd_byte))) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!sock->send_data(&ନିବେଶ_size, ଆକାର(ନିବେଶ_size))) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!sock->send_data(ନିବେଶ, ନିବେଶ_size)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
ସ୍ଥିର ତାର୍କିକ send_rpc_cmd(socket_ptr sock, ଗଣନା rpc_cmd cmd, ଅଚଳ ଶୂନ୍ୟ * ନିବେଶ, size_t ନିବେଶ_size, ଶୂନ୍ୟ * output, size_t output_size) {
    ଯଦି (!send_rpc_cmd(sock, cmd, ନିବେଶ, ନିବେଶ_size)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint64_t out_size;
    ଯଦି (!sock->recv_data(&out_size, ଆକାର(out_size))) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (out_size != output_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!sock->recv_data(output, output_size)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
ସ୍ଥିର constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rpc_cpu_relax() {
#ଯଦି defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ ଅସ୍ଥିର("yield" ::: "memory");
#ଅନ୍ୟଥା
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
ସ୍ଥିର ତାର୍କିକ negotiate_hello(ଅଚଳ std::shared_ptr<socket_t> & sock) {
    rpc_ସନ୍ଦେଶ_hello_req request = {};
    rpc_ସନ୍ଦେଶ_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    ତାର୍କିକ status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, ଆକାର(request), &response, ଆକାର(response));
    RPC_STATUS_ASSERT(status);

    ଯଦି (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

template <typename T>
ଶ୍ରେଣୀ message_queue {
ସର୍ବସାଧାରଣ:
    message_queue() {}

    ତାର୍କିକ ଠେଲ(ଅଚଳ T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        ଯଦି (interrupted) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        queue.ଠେଲ(value);
        cvar.notify_all();
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ କାଢ଼(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [ଏହି] { ଫେରନ୍ତୁ !queue.empty() || interrupted; });
        ଯଦି (interrupted) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        *out = queue.front();
        queue.କାଢ଼();
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ତାର୍କିକ try_କାଢ଼(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        ଯଦି (interrupted || queue.empty()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        *out = queue.front();
        queue.କାଢ଼();
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଶୂନ୍ୟ interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = ସତ୍ୟ;
        lock.unlock();
        cvar.notify_all();
    }

ବ୍ୟକ୍ତିଗତ:
    ତାର୍କିକ interrupted = ମିଥ୍ୟା;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

ଶ୍ରେଣୀ rpc_dispatcher {
ସର୍ବସାଧାରଣ:
    rpc_dispatcher() {
    }

    ଶୂନ୍ୟ send(ଗଣନା rpc_cmd cmd, std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ> ନିବେଶ, size_t ନିବେଶ_size);
    ଶୂନ୍ୟ send(ଗଣନା rpc_cmd cmd, std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ> ନିବେଶ, size_t ନିବେଶ_size, ଶୂନ୍ୟ * output, size_t output_size);
    ଶୂନ୍ୟ send_async(ଗଣନା rpc_cmd cmd, std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ> ନିବେଶ, size_t ନିବେଶ_size);
    ଶୂନ୍ୟ send_async(ଗଣନା rpc_cmd cmd, std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ> ନିବେଶ, size_t ନିବେଶ_size, ଶୂନ୍ୟ * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    ଶୂନ୍ୟ event_free(ggml_backend_event_t event);
    ଶୂନ୍ୟ event_synchronize(ggml_backend_event_t event);
    ଶୂନ୍ୟ event_record(ggml_backend_event_t event);
    ଶୂନ୍ୟ synchronize();
    ଶୂନ୍ୟ busy_spin_acquire();
    ଶୂନ୍ୟ busy_spin_release();
    ଶୂନ୍ୟ graph_compute(uint32_t device, ଅଚଳ ggml_cgraph * cgraph);

    ଶୂନ୍ୟ start(ଅଚଳ std::string & endpoint);
    ଶୂନ୍ୟ work();

    ~rpc_dispatcher();

ବ୍ୟକ୍ତିଗତ:
    ସଂରଚନା rpc_ସନ୍ଦେଶ {
        rpc_cmd                       cmd;
        std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ>   ନିବେଶ;
        size_t                        ନିବେଶ_size;
        ଶୂନ୍ୟ                        * output;
        size_t                        output_size;
        std::promise<ଶୂନ୍ୟ>            completion;
    };
    using rpc_ସନ୍ଦେଶ_ptr   = std::shared_ptr<rpc_ସନ୍ଦେଶ>;
    using rpc_ସନ୍ଦେଶ_queue = message_queue<rpc_ସନ୍ଦେଶ_ptr>;
    ସଂରଚନା rpc_event {
        rpc_ସନ୍ଦେଶ_ptr              ସନ୍ଦେଶ;
        std::shared_future<ଶୂନ୍ୟ> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_ସନ୍ଦେଶ_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

ସ୍ଥିର ଶୂନ୍ୟ rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

ଶୂନ୍ୟ rpc_dispatcher::send(ଗଣନା rpc_cmd cmd, std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ> ନିବେଶ, size_t ନିବେଶ_size) {
    auto ସନ୍ଦେଶ = std::make_shared<rpc_ସନ୍ଦେଶ>();
    ସନ୍ଦେଶ->cmd = cmd;
    ସନ୍ଦେଶ->ନିବେଶ = ନିବେଶ;
    ସନ୍ଦେଶ->ନିବେଶ_size = ନିବେଶ_size;
    ସନ୍ଦେଶ->output = nullptr;
    ସନ୍ଦେଶ->output_size = 0;
    GGML_ASSERT(queue.ଠେଲ(ସନ୍ଦେଶ));
    auto future = ସନ୍ଦେଶ->completion.get_future();
    future.wait();
}

ଶୂନ୍ୟ rpc_dispatcher::send_async(ଗଣନା rpc_cmd cmd, std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ> ନିବେଶ, size_t ନିବେଶ_size) {
    auto ସନ୍ଦେଶ = std::make_shared<rpc_ସନ୍ଦେଶ>();
    ସନ୍ଦେଶ->cmd = cmd;
    ସନ୍ଦେଶ->ନିବେଶ = ନିବେଶ;
    ସନ୍ଦେଶ->ନିବେଶ_size = ନିବେଶ_size;
    ସନ୍ଦେଶ->output = nullptr;
    ସନ୍ଦେଶ->output_size = 0;
    GGML_ASSERT(queue.ଠେଲ(ସନ୍ଦେଶ));
}

ଶୂନ୍ୟ rpc_dispatcher::send(ଗଣନା rpc_cmd cmd, std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ> ନିବେଶ, size_t ନିବେଶ_size, ଶୂନ୍ୟ * output, size_t output_size) {
    auto ସନ୍ଦେଶ = std::make_shared<rpc_ସନ୍ଦେଶ>();
    ସନ୍ଦେଶ->cmd = cmd;
    ସନ୍ଦେଶ->ନିବେଶ = ନିବେଶ;
    ସନ୍ଦେଶ->ନିବେଶ_size = ନିବେଶ_size;
    ସନ୍ଦେଶ->output = output;
    ସନ୍ଦେଶ->output_size = output_size;
    GGML_ASSERT(queue.ଠେଲ(ସନ୍ଦେଶ));
    auto future = ସନ୍ଦେଶ->completion.get_future();
    future.wait();
}

ଶୂନ୍ୟ rpc_dispatcher::send_async(ଗଣନା rpc_cmd cmd, std::shared_ptr<ଅଚଳ ଶୂନ୍ୟ> ନିବେଶ, size_t ନିବେଶ_size, ଶୂନ୍ୟ * output, size_t output_size) {
    auto ସନ୍ଦେଶ = std::make_shared<rpc_ସନ୍ଦେଶ>();
    ସନ୍ଦେଶ->cmd = cmd;
    ସନ୍ଦେଶ->ନିବେଶ = ନିବେଶ;
    ସନ୍ଦେଶ->ନିବେଶ_size = ନିବେଶ_size;
    ସନ୍ଦେଶ->output = output;
    ସନ୍ଦେଶ->output_size = output_size;
    GGML_ASSERT(queue.ଠେଲ(ସନ୍ଦେଶ));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = ନୂତନ rpc_event;
    ev->ସନ୍ଦେଶ = std::make_shared<rpc_ସନ୍ଦେଶ>();
    ev->ସନ୍ଦେଶ->cmd = RPC_CMD_NONE;
    ev->sf = ev->ସନ୍ଦେଶ->completion.get_future().share();
    GGML_ASSERT(queue.ଠେଲ(ev->ସନ୍ଦେଶ));
    ଫେରନ୍ତୁ ନୂତନ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

ଶୂନ୍ୟ rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

ଶୂନ୍ୟ rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

ଶୂନ୍ୟ rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->ସନ୍ଦେଶ = std::make_shared<rpc_ସନ୍ଦେଶ>();
    ev->ସନ୍ଦେଶ->cmd = RPC_CMD_NONE;
    ev->sf = ev->ସନ୍ଦେଶ->completion.get_future().share();
    GGML_ASSERT(queue.ଠେଲ(ev->ସନ୍ଦେଶ));
}

ଶୂନ୍ୟ rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto ସନ୍ଦେଶ = std::make_shared<rpc_ସନ୍ଦେଶ>();
    ସନ୍ଦେଶ->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ଠେଲ(ସନ୍ଦେଶ));
    ସନ୍ଦେଶ->completion.get_future().wait();
}

ଶୂନ୍ୟ rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

ଶୂନ୍ୟ rpc_dispatcher::busy_spin_release() {
    ଅଚଳ unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

ଶୂନ୍ୟ rpc_dispatcher::start(ଅଚଳ std::string & endpoint) {
    std::string host;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା port;
    ଯଦି (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    ଯଦି (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    ଯଦି (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    ଯଦି (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = ସତ୍ୟ;
    thread = std::thread(rpc_dispatcher_trampoline, ଏହି);
}

ଶୂନ୍ୟ rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    ଯେତେବେଳେ (running) {
        rpc_ସନ୍ଦେଶ_ptr ସନ୍ଦେଶ_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ଅଚଳ ତାର୍କିକ spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        ଯଦି (spin) {
            ଯଦି (!queue.try_କାଢ଼(&ସନ୍ଦେଶ_ptr)) {
                rpc_cpu_relax();
                ଜାରି;
            }
        } ଅନ୍ୟଥା ଯଦି (!queue.କାଢ଼(&ସନ୍ଦେଶ_ptr)) {
            ଭାଙ୍ଗ;
        }
        ଯଦି (ସନ୍ଦେଶ_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(ସନ୍ଦେଶ_ptr->cmd),
                     rpc_log_format_size(ସନ୍ଦେଶ_ptr->ନିବେଶ_size).c_str(), rpc_log_format_size(ସନ୍ଦେଶ_ptr->output_size).c_str());
            ଅଚଳ ତାର୍କିକ trace = rpc_log_debug_level() >= 3;
            ଅଚଳ auto t0 = std::chrono::steady_clock::now();
            ଯଦି (ସନ୍ଦେଶ_ptr->output) {
                ତାର୍କିକ status = send_rpc_cmd(sock, ସନ୍ଦେଶ_ptr->cmd, ସନ୍ଦେଶ_ptr->ନିବେଶ.get(), ସନ୍ଦେଶ_ptr->ନିବେଶ_size, ସନ୍ଦେଶ_ptr->output, ସନ୍ଦେଶ_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } ଅନ୍ୟଥା {
                ତାର୍କିକ status = send_rpc_cmd(sock, ସନ୍ଦେଶ_ptr->cmd, ସନ୍ଦେଶ_ptr->ନିବେଶ.get(), ସନ୍ଦେଶ_ptr->ନିବେଶ_size);
                RPC_STATUS_ASSERT(status);
            }
            ଯଦି (trace) {
                ଅଚଳ std::chrono::duration<ଦ୍ୱିଗୁଣ, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(ସନ୍ଦେଶ_ptr->cmd), elapsed.count());
            }
        } ଅନ୍ୟଥା {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        ସନ୍ଦେଶ_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = ମିଥ୍ୟା;
    queue.interrupt();
    sock = nullptr;
    ଯଦି (thread.joinable()) {
        thread.join();
    }
}

ସ୍ଥିର std::shared_ptr<rpc_dispatcher> get_dispatcher(ଅଚଳ std::string & endpoint) {
    ସ୍ଥିର std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    ସ୍ଥିର std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    ଯଦି (it != dispatchers.end()) {
        ଯଦି (auto dispatcher = it->second.lock()) {
            ଫେରନ୍ତୁ dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    ଫେରନ୍ତୁ dispatcher;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, ଆକାର(*request));
    delete ctx;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    ଯଦି (ctx->base_ptr != nullptr) {
        ଫେରନ୍ତୁ ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_ସନ୍ଦେଶ_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, ଆକାର(*request), &response, ଆକାର(response));
    ctx->base_ptr = reinterpret_cast<ଶୂନ୍ୟ *>(response.base_ptr);
    ଫେରନ୍ତୁ ctx->base_ptr;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    ଫେରନ୍ତୁ buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

ସ୍ଥିର rpc_tensor serialize_tensor(ଅଚଳ ggml_tensor * tensor, ଅଚଳ rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    ଯଦି (!tensor) {
        memset(&result, 0, ଆକାର(result));
        ଫେରନ୍ତୁ result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    ଯଦି (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        ଯଦି (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } ଅନ୍ୟଥା {
            result.buffer = 0;
            result.data = 0;
        }
    } ଅନ୍ୟଥା {
        result.buffer = 0;
        result.data   = 0;
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
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, ଆକାର(result.name));
    result.use_ଗଣନା = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    ଫେରନ୍ତୁ result;
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    ଯଦି (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, ଆକାର(*request));
    }
    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, ଆକାର(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
ସ୍ଥିର std::shared_ptr<uint8_t> serialize_set_tensor(ଅଚଳ rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, ଅଚଳ ଶୂନ୍ୟ * data, size_t size, size_t & ନିବେଶ_size) {
    ନିବେଶ_size = ଆକାର(rpc_tensor) + ଆକାର(cache_flag) + ଆକାର(offset) + size;
    uint8_t * ନିବେଶ = ନୂତନ uint8_t[ନିବେଶ_size]();
    uint8_t * p = ନିବେଶ;
    memcpy(p, &rpc_tensor, ଆକାର(rpc_tensor)); p += ଆକାର(rpc_tensor);
    memcpy(p, &cache_flag, ଆକାର(cache_flag)); p += ଆକାର(cache_flag);
    memcpy(p, &offset,     ଆକାର(offset));     p += ଆକାର(offset);
    memcpy(p, data, size);
    ଫେରନ୍ତୁ std::shared_ptr<uint8_t>(ନିବେଶ, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
ସ୍ଥିର ତାର୍କିକ rpc_use_hash_cache(ଅଚଳ ggml_tensor * tensor, size_t size) {
    ଫେରନ୍ତୁ size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    ଯଦି (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ଅଚଳ uint8_t*)data, size);
        rpc_ସନ୍ଦେଶ_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, ଆକାର(*request), &response, ଆକାର(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        ଯଦି (response.result) {
            // the server has the same data, no need to send it
            ଫେରନ୍ତୁ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ନିବେଶ_size;
    auto ନିବେଶ = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, ନିବେଶ_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, ନିବେଶ, ନିବେଶ_size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t ନିବେଶ_size = ଆକାର(rpc_tensor) + 4*ଆକାର(uint64_t) + size*n_copies;
    uint8_t * ନିବେଶ = ନୂତନ uint8_t[ନିବେଶ_size]();
    uint8_t * dest = ନିବେଶ;
    memcpy(dest, &rpc_tensor, ଆକାର(rpc_tensor));
    dest += ଆକାର(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, ଆକାର(header));
    dest += ଆକାର(header);
    ପାଇଁ (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (ଅଚଳ ଅକ୍ଷର *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> ନିବେଶ_ptr(ନିବେଶ, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, ନିବେଶ_ptr, ନିବେଶ_size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    ଯଦି (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, ଆକାର(*request), data, size*n_copies);
    } ଅନ୍ୟଥା {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, ଆକାର(*request), packed.data(), packed.size());
        ପାଇଁ (size_t i = 0; i < n_copies; i++) {
            memcpy((ଅକ୍ଷର *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, ଆକାର(*request), data, size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, ଅଚଳ ggml_tensor * src, ggml_tensor * dst) {
    ଯଦି (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        ଯଦି (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_ସନ୍ଦେଶ_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, ଆକାର(*request), &response, ଆକାର(response));
        ଫେରନ୍ତୁ response.result;
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, ଆକାର(*request));
}

ସ୍ଥିର ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ ଖାଲି,
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ଫେରନ୍ତୁ buft_ctx->name.c_str();
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_ସନ୍ଦେଶ_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, ଆକାର(*request), &response, ଆକାର(response));
    ଯଦି (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            ନୂତନ ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        ଫେରନ୍ତୁ buffer;
    } ଅନ୍ୟଥା {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        ଫେରନ୍ତୁ nullptr;
    }
}

ସ୍ଥିର size_t get_alignment(ଅଚଳ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_get_alignment_req>();
    request->device = device;
    rpc_ସନ୍ଦେଶ_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, ଆକାର(*request), &response, ଆକାର(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    ଫେରନ୍ତୁ response.alignment;
}

ସ୍ଥିର size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ଫେରନ୍ତୁ buft_ctx->alignment;
}

ସ୍ଥିର size_t get_max_size(ଅଚଳ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_get_max_size_req>();
    request->device = device;
    rpc_ସନ୍ଦେଶ_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, ଆକାର(*request), &response, ଆକାର(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    ଫେରନ୍ତୁ response.max_size;
}

ସ୍ଥିର size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ଫେରନ୍ତୁ buft_ctx->max_size;
}

ସ୍ଥିର size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    ତାର୍କିକ rpc_get = ମିଥ୍ୟା;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    ଯଦି (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ଅଚଳ size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        ସଂରଚନା alloc_size_cache_ଚାବି {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / ଆକାର(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_ଚାବି ଚାବି = {};
        ଚାବି.device = buft_ctx->device;
        ଚାବି.type = tensor->type;
        ଚାବି.op = tensor->op;
        memcpy(ଚାବି.op_params, tensor->op_params, ଆକାର(ଚାବି.op_params));
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
            ଚାବି.ne[i] = (uint32_t)tensor->ne[i];
            ଚାବି.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ଅଚଳ uint8_t *)&ଚାବି, ଆକାର(ଚାବି));
        cache_hash = fnv_hash((ଅଚଳ uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        ସ୍ଥିର std::mutex cache_mutex;
        ସ୍ଥିର std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            ଯଦି (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                ଫେରନ୍ତୁ std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_ସନ୍ଦେଶ_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, ଆକାର(*request), &response, ଆକାର(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        ଫେରନ୍ତୁ std::max<size_t>(response.alloc_size, min_size);
    }

    ଫେରନ୍ତୁ ggml_nbytes(tensor);
}

ସ୍ଥିର ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ଖାଲି,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ଖାଲି,
    /* .is_host             = */ ଖାଲି,
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    ଫେରନ୍ତୁ rpc_ctx->name.c_str();
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    ଯଦି (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((ଅଚଳ uint8_t*)data, size);
        rpc_ସନ୍ଦେଶ_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, ଆକାର(*request), &response, ଆକାର(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        ଯଦି (response.result) {
            // the server has the same data, no need to send it
            ଫେରନ୍ତୁ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ନିବେଶ_size;
    auto ନିବେଶ = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, ନିବେଶ_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, ନିବେଶ, ନିବେଶ_size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, ଆକାର(*request), data, size);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

ସ୍ଥିର ଶୂନ୍ୟ add_tensor(ggml_tensor * tensor, ଅଚଳ ggml_cgraph * cgraph, ଅଚଳ rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    ଯଦି (tensor == nullptr) {
        ଫେରନ୍ତୁ;
    }
    ଯଦି (visited.find(tensor) != visited.end()) {
        ଫେରନ୍ତୁ;
    }
    visited.insert(tensor);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    ଅଚଳ size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    ଯଦି (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_ଗଣନା = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

ସ୍ଥିର uint8_t * serialize_graph(uint32_t device, ଅଚଳ ggml_cgraph * cgraph, ଅଚଳ rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    ପାଇଁ (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*ଆକାର(uint32_t) + ଆକାର(uint64_t) + n_nodes * ଆକାର(uint64_t) + ଆକାର(uint32_t) + n_tensors * ଆକାର(rpc_tensor);
    uint8_t * output = ନୂତନ uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, ଆକାର(device));
    dest += ଆକାର(device);
    memcpy(dest, &cgraph->uid, ଆକାର(cgraph->uid));
    dest += ଆକାର(cgraph->uid);
    memcpy(dest, &n_nodes, ଆକାର(n_nodes));
    dest += ଆକାର(n_nodes);
    ପାଇଁ (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * ଆକାର(uint64_t), &cgraph->nodes[i], ଆକାର(uint64_t));
    }
    dest += n_nodes * ଆକାର(uint64_t);
    memcpy(dest, &n_tensors, ଆକାର(n_tensors));
    dest += ଆକାର(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * ଆକାର(rpc_tensor));
    ଫେରନ୍ତୁ output;
}

ଶୂନ୍ୟ rpc_dispatcher::graph_compute(uint32_t device, ଅଚଳ ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    ତାର୍କିକ reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    ଯଦି (reuse) {
        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, ଆକାର(*request));
    } ଅନ୍ୟଥା {
        ଯଦି (cgraph->uid != 0) {
            ଯଦି (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t ନିବେଶ_size = 0;
        uint8_t * ନିବେଶ = serialize_graph(device, cgraph, ଏହି, &ନିବେଶ_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(ନିବେଶ_size).c_str());
        std::shared_ptr<uint8_t> ନିବେଶ_ptr(ନିବେଶ, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, ନିବେଶ_ptr, ନିବେଶ_size);
    }
}

ସ୍ଥିର ଗଣନା ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ଅଚଳ ggml_tensor * tensor, ଶୂନ୍ୟ * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

ସ୍ଥିର ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ଖାଲି,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ ଖାଲି,
    /* .graph_plan_free         = */ ଖାଲି,
    /* .graph_plan_update       = */ ଖାଲି,
    /* .graph_plan_compute      = */ ଖାଲି,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ ଖାଲି,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(ଅଚଳ ଅକ୍ଷର * endpoint, uint32_t device) {
    ସ୍ଥିର std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    ସ୍ଥିର std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    ଯଦି (it != buft_map.end()) {
        ଫେରନ୍ତୁ it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = ନୂତନ ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = ନୂତନ ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    ଫେରନ୍ତୁ buft;
}

ggml_backend_t ggml_backend_rpc_init(ଅଚଳ ଅକ୍ଷର * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = ନୂତନ ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = ନୂତନ ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    ଫେରନ୍ତୁ backend;
}

ତାର୍କିକ ggml_backend_is_rpc(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend != ଖାଲି && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

ଶୂନ୍ୟ ggml_backend_rpc_get_device_memory(ଅଚଳ ଅକ୍ଷର * endpoint, uint32_t device, size_t * ମୁକ୍ତ, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_ସନ୍ଦେଶ_get_device_memory_req>();
    request->device = device;
    rpc_ସନ୍ଦେଶ_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, ଆକାର(*request), &response, ଆକାର(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *ମୁକ୍ତ = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

ଶ୍ରେଣୀ rpc_server {
ସର୍ବସାଧାରଣ:
    rpc_server(std::vector<ggml_backend_t> all_backends, ଅଚଳ ଅକ୍ଷର * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    ଶୂନ୍ୟ hello(rpc_ସନ୍ଦେଶ_hello_rsp & response);
    ତାର୍କିକ alloc_buffer(ଅଚଳ rpc_ସନ୍ଦେଶ_alloc_buffer_req & request, rpc_ସନ୍ଦେଶ_alloc_buffer_rsp & response);
    ତାର୍କିକ get_alignment(ଅଚଳ rpc_ସନ୍ଦେଶ_get_alignment_req & request, rpc_ସନ୍ଦେଶ_get_alignment_rsp & response);
    ତାର୍କିକ get_max_size(ଅଚଳ rpc_ସନ୍ଦେଶ_get_max_size_req & request, rpc_ସନ୍ଦେଶ_get_max_size_rsp & response);
    ତାର୍କିକ buffer_get_base(ଅଚଳ rpc_ସନ୍ଦେଶ_buffer_get_base_req & request, rpc_ସନ୍ଦେଶ_buffer_get_base_rsp & response);
    ତାର୍କିକ free_buffer(ଅଚଳ rpc_ସନ୍ଦେଶ_free_buffer_req & request);
    ତାର୍କିକ buffer_clear(ଅଚଳ rpc_ସନ୍ଦେଶ_buffer_clear_req & request);
    ତାର୍କିକ memset_tensor(ଅଚଳ rpc_ସନ୍ଦେଶ_memset_tensor_req & request);
    ତାର୍କିକ set_tensor(ଅଚଳ std::vector<uint8_t> & ନିବେଶ);
    ତାର୍କିକ set_tensor_2d(ଅଚଳ std::vector<uint8_t> & ନିବେଶ);
    ତାର୍କିକ set_tensor_hash(ଅଚଳ rpc_ସନ୍ଦେଶ_set_tensor_hash_req & request, rpc_ସନ୍ଦେଶ_set_tensor_hash_rsp & response);
    ତାର୍କିକ get_tensor(ଅଚଳ rpc_ସନ୍ଦେଶ_get_tensor_req & request, std::vector<uint8_t> & response);
    ତାର୍କିକ get_tensor_2d(ଅଚଳ rpc_ସନ୍ଦେଶ_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    ତାର୍କିକ copy_tensor(ଅଚଳ rpc_ସନ୍ଦେଶ_copy_tensor_req & request, rpc_ସନ୍ଦେଶ_copy_tensor_rsp & response);
    ତାର୍କିକ graph_compute(ଅଚଳ std::vector<uint8_t> & ନିବେଶ);
    ତାର୍କିକ graph_recompute(ଅଚଳ rpc_ସନ୍ଦେଶ_graph_recompute_req & request);
    ତାର୍କିକ comm_init(ଅଚଳ rpc_ସନ୍ଦେଶ_comm_init_req & request, rpc_ସନ୍ଦେଶ_comm_init_rsp & response);
    ତାର୍କିକ comm_allreduce(ଅଚଳ rpc_ସନ୍ଦେଶ_comm_allreduce_req & request);
    ତାର୍କିକ comm_free(ଅଚଳ rpc_ସନ୍ଦେଶ_comm_free_req & request);
    ତାର୍କିକ init_tensor(ଅଚଳ rpc_ସନ୍ଦେଶ_init_tensor_req & request);
    ତାର୍କିକ get_alloc_size(ଅଚଳ rpc_ସନ୍ଦେଶ_get_alloc_size_req & request, rpc_ସନ୍ଦେଶ_get_alloc_size_rsp & response);
    ତାର୍କିକ get_device_memory(ଅଚଳ rpc_ସନ୍ଦେଶ_get_device_memory_req & request, rpc_ସନ୍ଦେଶ_get_device_memory_rsp & response);

    ସଂରଚନା stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

ବ୍ୟକ୍ତିଗତ:
    ଶୂନ୍ୟ sync_all_backends();
    ତାର୍କିକ get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(ସଂରଚନା ggml_context * ctx, ଅଚଳ rpc_tensor * tensor);
    ggml_tensor * create_ଗଣ୍ଠି(uint64_t id,
                              ସଂରଚନା ggml_context * ctx,
                              ଅଚଳ std::unordered_map<uint64_t, ଅଚଳ rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, ସଂରଚନା ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    ସଂରଚନା comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ଅଚଳ ଅକ୍ଷର * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

ଶୂନ୍ୟ rpc_server::hello(rpc_ସନ୍ଦେଶ_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

ତାର୍କିକ rpc_server::get_alloc_size(ଅଚଳ rpc_ସନ୍ଦେଶ_get_alloc_size_req & request, rpc_ସନ୍ଦେଶ_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    ଯଦି (dev_id >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_backend_buffer_type_t buft;
    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ଯଦି (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        ଯଦି (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (ଶୂନ୍ୟ*)tensor->buffer, tensor->data);
    ଯଦି (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } ଅନ୍ୟଥା {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::alloc_buffer(ଅଚଳ rpc_ସନ୍ଦେଶ_alloc_buffer_req & request, rpc_ସନ୍ଦେଶ_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    ଯଦି (dev_id >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    ଯଦି (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } ଅନ୍ୟଥା {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::get_alignment(ଅଚଳ rpc_ସନ୍ଦେଶ_get_alignment_req & request, rpc_ସନ୍ଦେଶ_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    ଯଦି (dev_id >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::get_max_size(ଅଚଳ rpc_ସନ୍ଦେଶ_get_max_size_req & request, rpc_ସନ୍ଦେଶ_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    ଯଦି (dev_id >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::buffer_get_base(ଅଚଳ rpc_ସନ୍ଦେଶ_buffer_get_base_req & request, rpc_ସନ୍ଦେଶ_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    ଯଦି (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଶୂନ୍ୟ * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::free_buffer(ଅଚଳ rpc_ସନ୍ଦେଶ_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    ଯଦି (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    ପାଇଁ (auto & sgs : stored_graphs) {
        ପାଇଁ (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::buffer_clear(ଅଚଳ rpc_ସନ୍ଦେଶ_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    ଯଦି (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::memset_tensor(ଅଚଳ rpc_ସନ୍ଦେଶ_memset_tensor_req & request) {
    sync_all_backends();
    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ଯଦି (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint64_t tensor_size = ggml_nbytes(tensor);
    ଯଦି (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    ଅଚଳ uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    ଯଦି (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ uint64_t data_offset = request.tensor.data - buffer_start;
    ଯଦି (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ଶୂନ୍ୟ *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ggml_tensor * rpc_server::deserialize_tensor(ସଂରଚନା ggml_context * ctx, ଅଚଳ rpc_tensor * tensor) {
    // Validate tensor type before using it
    ଯଦି (tensor->type >= GGML_TYPE_ଗଣନା) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        ଫେରନ୍ତୁ nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    ଯଦି (ggml_blck_size((ଗଣନା ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        ଫେରନ୍ତୁ nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    ଯଦି (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        ଫେରନ୍ତୁ nullptr;
    }

    ପାଇଁ (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    ଯଦି (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    ଯଦି (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        ଯଦି (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            ଫେରନ୍ତୁ nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    ପାଇଁ (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / ଆକାର(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<ଶୂନ୍ୟ *>(tensor->data);
    ggml_set_name(result, tensor->name);
    ଫେରନ୍ତୁ result;
}


ତାର୍କିକ rpc_server::set_tensor(ଅଚଳ std::vector<uint8_t> & ନିବେଶ) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    ଅଚଳ size_t header_size = ଆକାର(rpc_tensor) + ଆକାର(cache_flag) + ଆକାର(offset);
    ଯଦି (ନିବେଶ.size() < header_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ rpc_tensor * in_tensor = (ଅଚଳ rpc_tensor *)ନିବେଶ.data();
    memcpy(&cache_flag, ନିବେଶ.data() + ଆକାର(rpc_tensor), ଆକାର(cache_flag));
    memcpy(&offset,     ନିବେଶ.data() + ଆକାର(rpc_tensor) + ଆକାର(cache_flag), ଆକାର(offset));
    ଅଚଳ size_t size = ନିବେଶ.size() - header_size;

    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    ଯଦି (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ଶୂନ୍ୟ*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        ଅଚଳ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ଅଚଳ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ଯଦି (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଅଚଳ ଶୂନ୍ୟ * data = ନିବେଶ.data() + header_size;
    ଯଦି (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ଅଚଳ uint8_t*)data, size);
        ଅକ୍ଷର hash_str[17];
        snprintf(hash_str, ଆକାର(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.ଲିଖ((ଅଚଳ ଅକ୍ଷର *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::set_tensor_2d(ଅଚଳ std::vector<uint8_t> & ନିବେଶ) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    ଯଦି (ନିବେଶ.size() < ଆକାର(rpc_tensor) + 4*ଆକାର(uint64_t)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ rpc_tensor * in_tensor = (ଅଚଳ rpc_tensor *)ନିବେଶ.data();
    uint64_t header[4];
    memcpy(header, ନିବେଶ.data() + ଆକାର(rpc_tensor), ଆକାର(header));
    ଅଚଳ uint64_t offset   = header[0];
    ଅଚଳ uint64_t size     = header[1];
    ଅଚଳ uint64_t n_copies = header[2];
    ଅଚଳ uint64_t stride   = header[3];

    ଅଚଳ uint64_t data_size = ନିବେଶ.size() - ଆକାର(rpc_tensor) - 4*ଆକାର(uint64_t);
    ଯଦି (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    ଯଦି (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ଶୂନ୍ୟ*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        ଯଦି (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ uint64_t span = (n_copies - 1)*stride + size;
        ଅଚଳ uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        ଅଚଳ uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ଯଦି (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଅଚଳ ଶୂନ୍ୟ * data = ନିବେଶ.data() + ଆକାର(rpc_tensor) + 4*ଆକାର(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    ଯଦି (!cache_dir) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅକ୍ଷର hash_str[17];
    snprintf(hash_str, ଆକାର(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    ଯଦି (!fs::exists(cache_file, ec)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.ପଠ((ଅକ୍ଷର *)data.data(), size);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::set_tensor_hash(ଅଚଳ rpc_ସନ୍ଦେଶ_set_tensor_hash_req & request, rpc_ସନ୍ଦେଶ_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ଯଦି (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    std::vector<uint8_t> cached_file;
    ଯଦି (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ଶୂନ୍ୟ*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ଅଚଳ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ଅଚଳ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ଯଦି (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::init_tensor(ଅଚଳ rpc_ସନ୍ଦେଶ_init_tensor_req & request) {
    sync_all_backends();
    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ଯଦି (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (ଶୂନ୍ୟ*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    ଯଦି (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } ଅନ୍ୟଥା {
        ଯଦି (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    ଯଦି (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::get_tensor(ଅଚଳ rpc_ସନ୍ଦେଶ_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ଯଦି (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (ଶୂନ୍ୟ*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        ଅଚଳ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        ଅଚଳ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ଯଦି (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::get_tensor_2d(ଅଚଳ rpc_ସନ୍ଦେଶ_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ଯଦି (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ଶୂନ୍ୟ*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        ଯଦି (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        ଅଚଳ uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        ଅଚଳ uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        ଯଦି (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::copy_tensor(ଅଚଳ rpc_ସନ୍ଦେଶ_copy_tensor_req & request, rpc_ସନ୍ଦେଶ_copy_tensor_rsp & response) {
    sync_all_backends();
    ସଂରଚନା ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    ଯଦି (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    ଯଦି (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ggml_tensor * rpc_server::create_ଗଣ୍ଠି(uint64_t id,
                                      ସଂରଚନା ggml_context * ctx,
                                      ଅଚଳ std::unordered_map<uint64_t, ଅଚଳ rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, ସଂରଚନା ggml_tensor*> & tensor_map) {
    ଯଦି (tensor_map.find(id) != tensor_map.end()) {
        ଫେରନ୍ତୁ tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    ଯଦି (it_ptr == tensor_ptrs.end()) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଅଚଳ rpc_tensor * tensor = it_ptr->second;

    ସଂରଚନା ggml_tensor * result = deserialize_tensor(ctx, tensor);
    ଯଦି (result == nullptr) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        ଫେରନ୍ତୁ nullptr;
    }
    tensor_map[id] = result;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        ଯଦି (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } ଅନ୍ୟଥା {
            result->src[i] = create_ଗଣ୍ଠି(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            ଯଦି (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                ଫେରନ୍ତୁ nullptr;
            }
        }
    }

    // Handle view_src similarly
    ଯଦି (tensor->view_src == 0) {
        result->view_src = nullptr;
    } ଅନ୍ୟଥା {
        result->view_src = create_ଗଣ୍ଠି(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        ଯଦି (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            ଫେରନ୍ତୁ nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    ଫେରନ୍ତୁ result;
}

ତାର୍କିକ rpc_server::graph_compute(ଅଚଳ std::vector<uint8_t> & ନିବେଶ) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    ଯଦି (ନିବେଶ.size() < 2*ଆକାର(uint32_t) + ଆକାର(uint64_t)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ uint8_t * src = ନିବେଶ.data();
    uint32_t device;
    memcpy(&device, src, ଆକାର(device));
    src += ଆକାର(device);
    ଯଦି (device >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint64_t uid;
    memcpy(&uid, src, ଆକାର(uid));
    src += ଆକାର(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, ଆକାର(n_nodes));
    src += ଆକାର(n_nodes);
    ଯଦି (ନିବେଶ.size() < 2*ଆକାର(uint32_t) + ଆକାର(uint64_t) + n_nodes*ଆକାର(uint64_t) + ଆକାର(uint32_t)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ uint64_t * nodes = (ଅଚଳ uint64_t *)src;
    src += n_nodes*ଆକାର(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, ଆକାର(n_tensors));
    src += ଆକାର(n_tensors);
    ଯଦି (ନିବେଶ.size() < 2*ଆକାର(uint32_t) + ଆକାର(uint64_t) + n_nodes*ଆକାର(uint64_t) + ଆକାର(uint32_t) + n_tensors*ଆକାର(rpc_tensor)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ rpc_tensor * tensors = (ଅଚଳ rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    ଯଦି (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, ମିଥ୍ୟା);
    ଯଦି (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    ସଂରଚନା ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ସଂରଚନା ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, ମିଥ୍ୟା);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ଅଚଳ rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    ପାଇଁ (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ଅଚଳ auto t_deserialize = std::chrono::steady_clock::now();
    ପାଇଁ (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], ଆକାର(id));
        graph->nodes[i] = create_ଗଣ୍ଠି(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        ଯଦି (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ଅଚଳ size_t hash_pos = ggml_hash_ଭର୍ତ୍ତି(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_ଗଣନା;
        }
    }
    ଅଚଳ std::chrono::duration<ଦ୍ୱିଗୁଣ, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::graph_recompute(ଅଚଳ rpc_ସନ୍ଦେଶ_graph_recompute_req & request) {
    uint32_t device = request.device;
    ଯଦି (device >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    auto it = stored_graphs[device].find(request.uid);
    ଯଦି (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
ଶୂନ୍ୟ rpc_server::sync_all_backends() {
    ପାଇଁ (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
ତାର୍କିକ rpc_server::comm_init(ଅଚଳ rpc_ସନ୍ଦେଶ_comm_init_req & request, rpc_ସନ୍ଦେଶ_comm_init_rsp & response) {
    response.ok = 0;
    ଯଦି (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    comm_state & state = comm_states[request.device];
    ଯଦି (state.peer != nullptr) {
        response.ok = 1;
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    ଯଦି (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        ଯଦି (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        state.peer = srv->accept();
        ଯଦି (state.peer == nullptr) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ଯଦି (!state.peer->recv_data(remote_caps, ଆକାର(remote_caps))) {
            state.peer = nullptr;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        state.peer->get_caps(local_caps);
        ଯଦି (!state.peer->send_data(local_caps, ଆକାର(local_caps))) {
            state.peer = nullptr;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        state.peer->update_caps(remote_caps);
    } ଅନ୍ୟଥା {
        ଅଚଳ std::string host(request.host, strnlen(request.host, ଆକାର(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            ଯଦି (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        ଯଦି (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        state.peer->get_caps(local_caps);
        ଯଦି (!state.peer->send_data(local_caps, ଆକାର(local_caps)) ||
            !state.peer->recv_data(remote_caps, ଆକାର(remote_caps))) {
            state.peer = nullptr;
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::comm_allreduce(ଅଚଳ rpc_ସନ୍ଦେଶ_comm_allreduce_req & request) {
    ଯଦି (request.device >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    comm_state & state = comm_states[request.device];
    ଯଦି (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, ମିଥ୍ୟା);
    ସଂରଚନା ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ଖାଲି,
        /*.no_alloc   =*/ ସତ୍ୟ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    ଯଦି (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ size_t  nbytes = ggml_nbytes(t_dst);
    ଅଚଳ int64_t ne     = ggml_nelements(t_dst);
    ଯଦି (nbytes == 0) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ଅଚଳ ତାର୍କିକ   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ଅଚଳ size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    ଅଚଳ size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    ଯଦି (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    ଅକ୍ଷର * scratch_base = (ଅକ୍ଷର *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        ଫେରନ୍ତୁ t;
    };
    auto new_cpy_ଗଣ୍ଠି = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ଫେରନ୍ତୁ t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, ମିଥ୍ୟା);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != nullptr ? 2 : 1;
        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = nullptr;
    ggml_tensor * t_wire_recv = nullptr;
    ଯଦି (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_ଗଣ୍ଠି(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } ଅନ୍ୟଥା {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    ଯଦି (state.rank == 0) {
        ଯଦି (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    } ଅନ୍ୟଥା {
        ଯଦି (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    ଯଦି (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_ଗଣ୍ଠି(t_wire_recv, t_peer);
    } ଅନ୍ୟଥା {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    ଯଦି (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } ଅନ୍ୟଥା {
        compute_nodes(t_red, nullptr);
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::comm_free(ଅଚଳ rpc_ସନ୍ଦେଶ_comm_free_req & request) {
    ଯଦି (request.device >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    comm_states[request.device] = comm_state();
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ rpc_server::get_device_memory(ଅଚଳ rpc_ସନ୍ଦେଶ_get_device_memory_req & request, rpc_ସନ୍ଦେଶ_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    ଯଦି (dev_id >= backends.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    size_t ମୁକ୍ତ, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &ମୁକ୍ତ, &total);
    response.free_mem = ମୁକ୍ତ;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

rpc_server::~rpc_server() {
    ପାଇଁ (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ rpc_serve_client(ଅଚଳ std::vector<ggml_backend_t> & backends, ଅଚଳ ଅକ୍ଷର * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    ଯଦି (!sock->recv_data(&cmd, 1)) {
        ଫେରନ୍ତୁ;
    }
    ଯଦି (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        ଫେରନ୍ତୁ;
    }

    // Read input_size and validate protocol version
    uint64_t hello_ନିବେଶ_size;
    ଯଦି (!sock->recv_data(&hello_ନିବେଶ_size, ଆକାର(hello_ନିବେଶ_size))) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (hello_ନିବେଶ_size != ଆକାର(rpc_ସନ୍ଦେଶ_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_ନିବେଶ_size, ଆକାର(rpc_ସନ୍ଦେଶ_hello_req), RPC_PROTO_MAJOR_VERSION);
        ଫେରନ୍ତୁ;
    }

    rpc_ସନ୍ଦେଶ_hello_req req = {};
    ଯଦି (!sock->recv_data(&req, ଆକାର(req))) {
        ଫେରନ୍ତୁ;
    }

    rpc_ସନ୍ଦେଶ_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    ଯଦି (!send_ସନ୍ଦେଶ(sock, &rsp, ଆକାର(rsp))) {
        ଫେରନ୍ତୁ;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    ଯେତେବେଳେ (ସତ୍ୟ) {
        ଯଦି (!sock->recv_data(&cmd, 1)) {
            ଭାଙ୍ଗ;
        }
        ଯଦି (cmd >= RPC_CMD_ଗଣନା) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            ଭାଙ୍ଗ;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((ଗଣନା rpc_cmd)cmd));
        ବାଛ (cmd) {
            ସ୍ଥିତି RPC_CMD_HELLO: {
                // HELLO command is handled above
                ଫେରନ୍ତୁ;
            }
            ସ୍ଥିତି RPC_CMD_DEVICE_ଗଣନା: {
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, nullptr, 0)) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_device_ଗଣନା_rsp response;
                response.device_ଗଣନା = backends.size();
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_ALLOC_BUFFER: {
                rpc_ସନ୍ଦେଶ_alloc_buffer_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_alloc_buffer_rsp response;
                ଯଦି (!server.alloc_buffer(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_GET_ALLOC_SIZE: {
                rpc_ସନ୍ଦେଶ_get_alloc_size_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_get_alloc_size_rsp response;
                ଯଦି (!server.get_alloc_size(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_GET_ALIGNMENT: {
                rpc_ସନ୍ଦେଶ_get_alignment_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_get_alignment_rsp response;
                ଯଦି (!server.get_alignment(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_GET_MAX_SIZE: {
                rpc_ସନ୍ଦେଶ_get_max_size_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_get_max_size_rsp response;
                ଯଦି (!server.get_max_size(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_BUFFER_GET_BASE: {
                rpc_ସନ୍ଦେଶ_buffer_get_base_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_buffer_get_base_rsp response;
                ଯଦି (!server.buffer_get_base(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_FREE_BUFFER: {
                rpc_ସନ୍ଦେଶ_free_buffer_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.free_buffer(request)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_BUFFER_CLEAR: {
                rpc_ସନ୍ଦେଶ_buffer_clear_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.buffer_clear(request)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_MEMSET_TENSOR: {
                rpc_ସନ୍ଦେଶ_memset_tensor_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.memset_tensor(request)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> ନିବେଶ;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, ନିବେଶ)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.set_tensor(ନିବେଶ)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> ନିବେଶ;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, ନିବେଶ)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.set_tensor_2d(ନିବେଶ)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_GET_TENSOR_2D: {
                rpc_ସନ୍ଦେଶ_get_tensor_2d_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                std::vector<uint8_t> response;
                ଯଦି (!server.get_tensor_2d(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, response.data(), response.size())) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_SET_TENSOR_HASH: {
                rpc_ସନ୍ଦେଶ_set_tensor_hash_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_set_tensor_hash_rsp response;
                ଯଦି (!server.set_tensor_hash(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_INIT_TENSOR: {
                rpc_ସନ୍ଦେଶ_init_tensor_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request,ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.init_tensor(request)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_GET_TENSOR: {
                rpc_ସନ୍ଦେଶ_get_tensor_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                std::vector<uint8_t> response;
                ଯଦି (!server.get_tensor(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, response.data(), response.size())) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_COPY_TENSOR: {
                rpc_ସନ୍ଦେଶ_copy_tensor_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_copy_tensor_rsp response;
                ଯଦି (!server.copy_tensor(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> ନିବେଶ;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, ନିବେଶ)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.graph_compute(ନିବେଶ)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_ସନ୍ଦେଶ_graph_recompute_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.graph_recompute(request)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_COMM_INIT: {
                rpc_ସନ୍ଦେଶ_comm_init_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_comm_init_rsp response;
                ଯଦି (!server.comm_init(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_COMM_ALLREDUCE: {
                rpc_ସନ୍ଦେଶ_comm_allreduce_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.comm_allreduce(request)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_COMM_FREE: {
                rpc_ସନ୍ଦେଶ_comm_free_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!server.comm_free(request)) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_ସନ୍ଦେଶ_get_device_memory_req request;
                ଯଦି (!recv_ସନ୍ଦେଶ(sock, &request, ଆକାର(request))) {
                    ଫେରନ୍ତୁ;
                }
                rpc_ସନ୍ଦେଶ_get_device_memory_rsp response;
                ଯଦି (!server.get_device_memory(request, response)) {
                    ଫେରନ୍ତୁ;
                }
                ଯଦି (!send_ସନ୍ଦେଶ(sock, &response, ଆକାର(response))) {
                    ଫେରନ୍ତୁ;
                }
                ଭାଙ୍ଗ;
            }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((ଗଣନା rpc_cmd)cmd));
                ଫେରନ୍ତୁ;
            }
        }
    }
}

ଶୂନ୍ୟ ggml_backend_rpc_start_server(ଅଚଳ ଅକ୍ଷର * endpoint, ଅଚଳ ଅକ୍ଷର * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    ଯଦି (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        ଫେରନ୍ତୁ;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    ପାଇଁ (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t ମୁକ୍ତ, total;
        ggml_backend_dev_memory(dev, &ମୁକ୍ତ, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, ମୁକ୍ତ / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        ଯଦି (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            ଫେରନ୍ତୁ;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        ଯଦି (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            ଯଦି (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା port;
    ଯଦି (!parse_endpoint(endpoint, host, port)) {
        ଫେରନ୍ତୁ;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#ଅନ୍ୟଥା
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    ଯଦି (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        ଫେରନ୍ତୁ;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    ଯଦି (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        ଫେରନ୍ତୁ;
    }
    ଯେତେବେଳେ (ସତ୍ୟ) {
        auto client_socket = server_socket->accept();
        ଯଦି (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            ଫେରନ୍ତୁ;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    ପାଇଁ (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ଫେରନ୍ତୁ ctx->name.c_str();
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ଫେରନ୍ତୁ ctx->description.c_str();
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, ମୁକ୍ତ, total);
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, ସଂରଚନା ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ ସତ୍ୟ,
        /* .host_buffer           = */ ମିଥ୍ୟା,
        /* .buffer_from_host_ptr  = */ ମିଥ୍ୟା,
        /* .events                = */ ସତ୍ୟ,
        /* .mmap_support          = */ ସତ୍ୟ,
    };
}

ସ୍ଥିର ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ଫେରନ୍ତୁ ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ଫେରନ୍ତୁ ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ସଂରଚନା ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ଯଦି (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    ଫେରନ୍ତୁ buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

ସ୍ଥିର ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    ଫେରନ୍ତୁ dispatcher->event_new(dev);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ ଖାଲି,
    /* .buffer_from_host_ptr = */ ଖାଲି,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ ଖାଲି,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

ସଂରଚନା ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ଫେରନ୍ତୁ ctx ? ctx->name.c_str() : "RPC";
}

ସ୍ଥିର size_t ggml_backend_rpc_reg_get_device_ଗଣନା(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ଫେରନ୍ତୁ ctx ? ctx->devices.size() : 0;
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ଯଦି (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } ଅନ୍ୟଥା {
        GGML_ASSERT(index < ctx->devices.size());
        ଫେରନ୍ତୁ ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
ସଂରଚନା ggml_backend_rpc_comm_context {
    ସଂରଚନା rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_rpc_comm_free(ଶୂନ୍ୟ * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ଯଦି (comm_ctx == nullptr) {
        ଫେରନ୍ତୁ;
    }
    ପାଇଁ (ଅଚଳ auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, ଆକାର(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ଯଦି (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        ଫେରନ୍ତୁ nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        ଯଦି (!ggml_backend_is_rpc(backends[i])) {
            ଫେରନ୍ତୁ nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        ପାଇଁ (ଅଚଳ auto & rank : ranks) {
            ଯଦି (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                ଫେରନ୍ତୁ nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା port0;
    ଯଦି (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଅଚଳ uint32_t comm_port = (uint32_t) port0 + 1000;
    ଯଦି (host0.size() >= 64) {
        ଫେରନ୍ତୁ nullptr;
    }

    ପାଇଁ (ଅଚଳ auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_ସନ୍ଦେଶ_comm_init_rsp> responses(n_backends);
    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        ଯଦି (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, ଆକାର(*request), &responses[i], ଆକାର(responses[i]));
    }
    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    ତାର୍କିକ ok = ସତ୍ୟ;
    ପାଇଁ (size_t i = 0; i < n_backends; i++) {
        ଯଦି (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = ମିଥ୍ୟା;
        }
    }
    ଯଦି (!ok) {
        ପାଇଁ (ଅଚଳ auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        ଫେରନ୍ତୁ nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    ଫେରନ୍ତୁ ନୂତନ ggml_backend_rpc_comm_context{std::move(ranks)};
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_rpc_comm_allreduce_tensor(ଶୂନ୍ୟ * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ଯଦି (comm_ctx == nullptr) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ size_t n_ranks = comm_ctx->ranks.size();
    ଅଚଳ int64_t ne = ggml_nelements(tensors[0]);
    ଯଦି (ne == 0) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ପାଇଁ (size_t i = 0; i < n_ranks; i++) {
        ଯଦି (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        ଯଦି ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ପାଇଁ (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_ସନ୍ଦେଶ_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, ଆକାର(*request));
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, ଅଚଳ ଅକ୍ଷର * name) {
    ଯଦି (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_rpc_add_server;
    }
    ଯଦି (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_rpc_start_server;
    }
    ଯଦି (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_rpc_comm_init;
    }
    ଯଦି (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_rpc_comm_free;
    }
    ଯଦି (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    ଫେରନ୍ତୁ ଖାଲି;

    GGML_UNUSED(reg);
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_ଗଣନା,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(ଶୂନ୍ୟ) {
    ସ୍ଥିର ସଂରଚନା ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ ଖାଲି,
    };

    ଫେରନ୍ତୁ &ggml_backend_rpc_reg;
}

ସ୍ଥିର uint32_t ggml_backend_rpc_get_device_ଗଣନା(ଅଚଳ ଅକ୍ଷର * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_ସନ୍ଦେଶ_device_ଗଣନା_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_ଗଣନା, nullptr, 0, &response, ଆକାର(response));
    ଫେରନ୍ତୁ response.device_ଗଣନା;
}

ସ୍ଥିର ଅଚଳ ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_ଗଣନା,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(ଅଚଳ ଅକ୍ଷର * endpoint) {
    ସ୍ଥିର std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    ସ୍ଥିର std::mutex mutex;
    ସ୍ଥିର uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    ଯଦି (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        ଫେରନ୍ତୁ reg_map[endpoint];
    }
    uint32_t dev_ଗଣନା = ggml_backend_rpc_get_device_ଗଣନା(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_ଗଣନା);
    ଯଦି (dev_ଗଣନା == 0) {
        ଫେରନ୍ତୁ nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = ନୂତନ ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    ପାଇଁ (uint32_t ind = 0; ind < dev_ଗଣନା; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = ନୂତନ ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = ନୂତନ ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = ନୂତନ ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    ଫେରନ୍ତୁ reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

#അന്തര്ഭാവ "ggml-rpc.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-cpp.h"
#അന്തര്ഭാവ "transport.h"
#അന്തര്ഭാവ "log.h"

#അന്തര്ഭാവ <array>
#അന്തര്ഭാവ <chrono>
#അന്തര്ഭാവ <cinttypes>
#അന്തര്ഭാവ <optional>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <thread>
#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <queue>
#അന്തര്ഭാവ <condition_variable>
#അന്തര്ഭാവ <future>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <unordered_map>
#അന്തര്ഭാവ <unordered_set>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <filesystem>
#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <atomic>

നാമാകാശ fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) എങ്കിൽ (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(തള്ളുക, 1)
// ggml_tensor is serialized into rpc_tensor
ഘടന rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / വലുപ്പം(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    അക്ഷരം name[GGML_MAX_NAME];

    int32_t use_എണ്ണം;
};

static_assert(വലുപ്പം(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
എണ്ണൽ rpc_cmd {
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
    RPC_CMD_DEVICE_എണ്ണം,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_എണ്ണം,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

സ്ഥിതം സ്ഥിരം അക്ഷരം * rpc_cmd_name(എണ്ണൽ rpc_cmd cmd) {
    തിരഞ്ഞെടുപ്പ് (cmd) {
        അവസ്ഥ RPC_CMD_ALLOC_BUFFER:      തിരികെ "ALLOC_BUFFER";
        അവസ്ഥ RPC_CMD_GET_ALIGNMENT:     തിരികെ "GET_ALIGNMENT";
        അവസ്ഥ RPC_CMD_GET_MAX_SIZE:      തിരികെ "GET_MAX_SIZE";
        അവസ്ഥ RPC_CMD_BUFFER_GET_BASE:   തിരികെ "BUFFER_GET_BASE";
        അവസ്ഥ RPC_CMD_FREE_BUFFER:       തിരികെ "FREE_BUFFER";
        അവസ്ഥ RPC_CMD_BUFFER_CLEAR:      തിരികെ "BUFFER_CLEAR";
        അവസ്ഥ RPC_CMD_SET_TENSOR:        തിരികെ "SET_TENSOR";
        അവസ്ഥ RPC_CMD_SET_TENSOR_HASH:   തിരികെ "SET_TENSOR_HASH";
        അവസ്ഥ RPC_CMD_GET_TENSOR:        തിരികെ "GET_TENSOR";
        അവസ്ഥ RPC_CMD_COPY_TENSOR:       തിരികെ "COPY_TENSOR";
        അവസ്ഥ RPC_CMD_GRAPH_COMPUTE:     തിരികെ "GRAPH_COMPUTE";
        അവസ്ഥ RPC_CMD_GET_DEVICE_MEMORY: തിരികെ "GET_DEVICE_MEMORY";
        അവസ്ഥ RPC_CMD_INIT_TENSOR:       തിരികെ "INIT_TENSOR";
        അവസ്ഥ RPC_CMD_GET_ALLOC_SIZE:    തിരികെ "GET_ALLOC_SIZE";
        അവസ്ഥ RPC_CMD_HELLO:             തിരികെ "HELLO";
        അവസ്ഥ RPC_CMD_DEVICE_എണ്ണം:      തിരികെ "DEVICE_COUNT";
        അവസ്ഥ RPC_CMD_GRAPH_RECOMPUTE:   തിരികെ "GRAPH_RECOMPUTE";
        അവസ്ഥ RPC_CMD_MEMSET_TENSOR:     തിരികെ "MEMSET_TENSOR";
        അവസ്ഥ RPC_CMD_SET_TENSOR_2D:     തിരികെ "SET_TENSOR_2D";
        അവസ്ഥ RPC_CMD_GET_TENSOR_2D:     തിരികെ "GET_TENSOR_2D";
        അവസ്ഥ RPC_CMD_COMM_INIT:         തിരികെ "COMM_INIT";
        അവസ്ഥ RPC_CMD_COMM_ALLREDUCE:    തിരികെ "COMM_ALLREDUCE";
        അവസ്ഥ RPC_CMD_COMM_FREE:         തിരികെ "COMM_FREE";
        അവസ്ഥ RPC_CMD_NONE:              തിരികെ "NONE";
        സ്ഥിരസ്ഥിതി:                        തിരികെ "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
സ്ഥിരം size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
സ്ഥിരം size_t GRAPH_CACHE_MAX = 1024;

ഘടന rpc_സന്ദേശം_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

ഘടന rpc_സന്ദേശം_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

ഘടന rpc_സന്ദേശം_device_എണ്ണം_rsp {
    uint32_t device_എണ്ണം;
};

ഘടന rpc_സന്ദേശം_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

ഘടന rpc_സന്ദേശം_get_alloc_size_rsp {
    uint64_t alloc_size;
};

ഘടന rpc_സന്ദേശം_init_tensor_req {
    rpc_tensor tensor;
};

ഘടന rpc_സന്ദേശം_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

ഘടന rpc_സന്ദേശം_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

ഘടന rpc_സന്ദേശം_get_alignment_req {
    uint32_t device;
};

ഘടന rpc_സന്ദേശം_get_alignment_rsp {
    uint64_t alignment;
};

ഘടന rpc_സന്ദേശം_get_max_size_req {
    uint32_t device;
};

ഘടന rpc_സന്ദേശം_get_max_size_rsp {
    uint64_t max_size;
};

ഘടന rpc_സന്ദേശം_buffer_get_base_req {
    uint64_t remote_ptr;
};

ഘടന rpc_സന്ദേശം_buffer_get_base_rsp {
    uint64_t base_ptr;
};

ഘടന rpc_സന്ദേശം_free_buffer_req {
    uint64_t remote_ptr;
};

ഘടന rpc_സന്ദേശം_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

ഘടന rpc_സന്ദേശം_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

ഘടന rpc_സന്ദേശം_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

ഘടന rpc_സന്ദേശം_set_tensor_hash_rsp {
    uint8_t result;
};

ഘടന rpc_സന്ദേശം_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

ഘടന rpc_സന്ദേശം_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

ഘടന rpc_സന്ദേശം_copy_tensor_rsp {
    uint8_t result;
};

ഘടന rpc_സന്ദേശം_get_device_memory_req {
    uint32_t device;
};

ഘടന rpc_സന്ദേശം_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

ഘടന rpc_സന്ദേശം_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

ഘടന rpc_സന്ദേശം_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

ഘടന rpc_സന്ദേശം_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    അക്ഷരം     host[64];  // rank > 0: rank 0's host
};

ഘടന rpc_സന്ദേശം_comm_init_rsp {
    uint8_t ok;
};

ഘടന rpc_സന്ദേശം_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

ഘടന rpc_സന്ദേശം_comm_free_req {
    uint32_t device;
};

#pragma pack(എടുക്കുക)

// RPC data structures

സ്ഥിതം ggml_guid_t ggml_backend_rpc_guid() {
    സ്ഥിതം ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    തിരികെ &guid;
}

ഘടന ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

ഘടന ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

വർഗ്ഗം rpc_dispatcher;
ഘടന ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

ഘടന ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    ശൂന്യം                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
സ്ഥിതം uint64_t fnv_hash(സ്ഥിരം uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    സ്ഥിരം uint64_t fnv_അഭാജ്യം = 0x100000001b3ULL;

    വേണ്ടി (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_അഭാജ്യം;
    }
    തിരികെ hash;
}

സ്ഥിതം തർക്കം send_സന്ദേശം(socket_ptr sock, സ്ഥിരം ശൂന്യം * സന്ദേശം, size_t സന്ദേശം_size) {
    എങ്കിൽ (!sock->send_data(&സന്ദേശം_size, വലുപ്പം(സന്ദേശം_size))) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!sock->send_data(സന്ദേശം, സന്ദേശം_size)) {
        തിരികെ അസത്യം;
    }
    തിരികെ sock->flush();
}

സ്ഥിതം തർക്കം recv_സന്ദേശം(socket_ptr sock, ശൂന്യം * സന്ദേശം, size_t സന്ദേശം_size) {
    uint64_t size;
    എങ്കിൽ (!sock->recv_data(&size, വലുപ്പം(size))) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (size != സന്ദേശം_size) {
        തിരികെ അസത്യം;
    }
    തിരികെ sock->recv_data(സന്ദേശം, സന്ദേശം_size);
}

സ്ഥിതം തർക്കം recv_സന്ദേശം(socket_ptr sock, std::vector<uint8_t> & നിവേശം) {
    uint64_t size;
    എങ്കിൽ (!sock->recv_data(&size, വലുപ്പം(size))) {
        തിരികെ അസത്യം;
    }
    ശ്രമിക്കുക {
        നിവേശം.resize(size);
    } പിടിക്കുക (സ്ഥിരം std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        തിരികെ അസത്യം;
    }
    തിരികെ sock->recv_data(നിവേശം.data(), size);
}

സ്ഥിതം തർക്കം parse_endpoint(സ്ഥിരം std::string & endpoint, std::string & host, പൂർണ്ണസംഖ്യ & port) {
    size_t pos = endpoint.find(':');
    എങ്കിൽ (pos == std::string::npos) {
        തിരികെ അസത്യം;
    }
    host = endpoint.substr(0, pos);
    ശ്രമിക്കുക {
        port = std::stoi(endpoint.substr(pos + 1));
    } പിടിക്കുക (...) {
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
സ്ഥിതം തർക്കം send_rpc_cmd(socket_ptr sock, എണ്ണൽ rpc_cmd cmd, സ്ഥിരം ശൂന്യം * നിവേശം, size_t നിവേശം_size) {
    uint8_t cmd_byte = cmd;
    എങ്കിൽ (!sock->send_data(&cmd_byte, വലുപ്പം(cmd_byte))) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!sock->send_data(&നിവേശം_size, വലുപ്പം(നിവേശം_size))) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!sock->send_data(നിവേശം, നിവേശം_size)) {
        തിരികെ അസത്യം;
    }
    തിരികെ sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
സ്ഥിതം തർക്കം send_rpc_cmd(socket_ptr sock, എണ്ണൽ rpc_cmd cmd, സ്ഥിരം ശൂന്യം * നിവേശം, size_t നിവേശം_size, ശൂന്യം * output, size_t output_size) {
    എങ്കിൽ (!send_rpc_cmd(sock, cmd, നിവേശം, നിവേശം_size)) {
        തിരികെ അസത്യം;
    }
    uint64_t out_size;
    എങ്കിൽ (!sock->recv_data(&out_size, വലുപ്പം(out_size))) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (out_size != output_size) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!sock->recv_data(output, output_size)) {
        തിരികെ അസത്യം;
    }
    തിരികെ സത്യം;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
സ്ഥിതം constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

സ്ഥിതം വരിയിൽ ശൂന്യം rpc_cpu_relax() {
#എങ്കിൽ defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ അസ്ഥിര("yield" ::: "memory");
#അല്ലെങ്കിൽ
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
സ്ഥിതം തർക്കം negotiate_hello(സ്ഥിരം std::shared_ptr<socket_t> & sock) {
    rpc_സന്ദേശം_hello_req request = {};
    rpc_സന്ദേശം_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    തർക്കം status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, വലുപ്പം(request), &response, വലുപ്പം(response));
    RPC_STATUS_ASSERT(status);

    എങ്കിൽ (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        തിരികെ അസത്യം;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    തിരികെ സത്യം;
}

template <typename T>
വർഗ്ഗം message_queue {
പൊതു:
    message_queue() {}

    തർക്കം തള്ളുക(സ്ഥിരം T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        എങ്കിൽ (interrupted) {
            തിരികെ അസത്യം;
        }
        queue.തള്ളുക(value);
        cvar.notify_all();
        തിരികെ സത്യം;
    }

    തർക്കം എടുക്കുക(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [ഇത്] { തിരികെ !queue.empty() || interrupted; });
        എങ്കിൽ (interrupted) {
            തിരികെ അസത്യം;
        }
        *out = queue.front();
        queue.എടുക്കുക();
        തിരികെ സത്യം;
    }

    തർക്കം try_എടുക്കുക(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        എങ്കിൽ (interrupted || queue.empty()) {
            തിരികെ അസത്യം;
        }
        *out = queue.front();
        queue.എടുക്കുക();
        തിരികെ സത്യം;
    }

    ശൂന്യം interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = സത്യം;
        lock.unlock();
        cvar.notify_all();
    }

സ്വകാര്യ:
    തർക്കം interrupted = അസത്യം;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

വർഗ്ഗം rpc_dispatcher {
പൊതു:
    rpc_dispatcher() {
    }

    ശൂന്യം send(എണ്ണൽ rpc_cmd cmd, std::shared_ptr<സ്ഥിരം ശൂന്യം> നിവേശം, size_t നിവേശം_size);
    ശൂന്യം send(എണ്ണൽ rpc_cmd cmd, std::shared_ptr<സ്ഥിരം ശൂന്യം> നിവേശം, size_t നിവേശം_size, ശൂന്യം * output, size_t output_size);
    ശൂന്യം send_async(എണ്ണൽ rpc_cmd cmd, std::shared_ptr<സ്ഥിരം ശൂന്യം> നിവേശം, size_t നിവേശം_size);
    ശൂന്യം send_async(എണ്ണൽ rpc_cmd cmd, std::shared_ptr<സ്ഥിരം ശൂന്യം> നിവേശം, size_t നിവേശം_size, ശൂന്യം * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    ശൂന്യം event_free(ggml_backend_event_t event);
    ശൂന്യം event_synchronize(ggml_backend_event_t event);
    ശൂന്യം event_record(ggml_backend_event_t event);
    ശൂന്യം synchronize();
    ശൂന്യം busy_spin_acquire();
    ശൂന്യം busy_spin_release();
    ശൂന്യം graph_compute(uint32_t device, സ്ഥിരം ggml_cgraph * cgraph);

    ശൂന്യം start(സ്ഥിരം std::string & endpoint);
    ശൂന്യം work();

    ~rpc_dispatcher();

സ്വകാര്യ:
    ഘടന rpc_സന്ദേശം {
        rpc_cmd                       cmd;
        std::shared_ptr<സ്ഥിരം ശൂന്യം>   നിവേശം;
        size_t                        നിവേശം_size;
        ശൂന്യം                        * output;
        size_t                        output_size;
        std::promise<ശൂന്യം>            completion;
    };
    using rpc_സന്ദേശം_ptr   = std::shared_ptr<rpc_സന്ദേശം>;
    using rpc_സന്ദേശം_queue = message_queue<rpc_സന്ദേശം_ptr>;
    ഘടന rpc_event {
        rpc_സന്ദേശം_ptr              സന്ദേശം;
        std::shared_future<ശൂന്യം> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_സന്ദേശം_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

സ്ഥിതം ശൂന്യം rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

ശൂന്യം rpc_dispatcher::send(എണ്ണൽ rpc_cmd cmd, std::shared_ptr<സ്ഥിരം ശൂന്യം> നിവേശം, size_t നിവേശം_size) {
    auto സന്ദേശം = std::make_shared<rpc_സന്ദേശം>();
    സന്ദേശം->cmd = cmd;
    സന്ദേശം->നിവേശം = നിവേശം;
    സന്ദേശം->നിവേശം_size = നിവേശം_size;
    സന്ദേശം->output = nullptr;
    സന്ദേശം->output_size = 0;
    GGML_ASSERT(queue.തള്ളുക(സന്ദേശം));
    auto future = സന്ദേശം->completion.get_future();
    future.wait();
}

ശൂന്യം rpc_dispatcher::send_async(എണ്ണൽ rpc_cmd cmd, std::shared_ptr<സ്ഥിരം ശൂന്യം> നിവേശം, size_t നിവേശം_size) {
    auto സന്ദേശം = std::make_shared<rpc_സന്ദേശം>();
    സന്ദേശം->cmd = cmd;
    സന്ദേശം->നിവേശം = നിവേശം;
    സന്ദേശം->നിവേശം_size = നിവേശം_size;
    സന്ദേശം->output = nullptr;
    സന്ദേശം->output_size = 0;
    GGML_ASSERT(queue.തള്ളുക(സന്ദേശം));
}

ശൂന്യം rpc_dispatcher::send(എണ്ണൽ rpc_cmd cmd, std::shared_ptr<സ്ഥിരം ശൂന്യം> നിവേശം, size_t നിവേശം_size, ശൂന്യം * output, size_t output_size) {
    auto സന്ദേശം = std::make_shared<rpc_സന്ദേശം>();
    സന്ദേശം->cmd = cmd;
    സന്ദേശം->നിവേശം = നിവേശം;
    സന്ദേശം->നിവേശം_size = നിവേശം_size;
    സന്ദേശം->output = output;
    സന്ദേശം->output_size = output_size;
    GGML_ASSERT(queue.തള്ളുക(സന്ദേശം));
    auto future = സന്ദേശം->completion.get_future();
    future.wait();
}

ശൂന്യം rpc_dispatcher::send_async(എണ്ണൽ rpc_cmd cmd, std::shared_ptr<സ്ഥിരം ശൂന്യം> നിവേശം, size_t നിവേശം_size, ശൂന്യം * output, size_t output_size) {
    auto സന്ദേശം = std::make_shared<rpc_സന്ദേശം>();
    സന്ദേശം->cmd = cmd;
    സന്ദേശം->നിവേശം = നിവേശം;
    സന്ദേശം->നിവേശം_size = നിവേശം_size;
    സന്ദേശം->output = output;
    സന്ദേശം->output_size = output_size;
    GGML_ASSERT(queue.തള്ളുക(സന്ദേശം));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = പുതിയ rpc_event;
    ev->സന്ദേശം = std::make_shared<rpc_സന്ദേശം>();
    ev->സന്ദേശം->cmd = RPC_CMD_NONE;
    ev->sf = ev->സന്ദേശം->completion.get_future().share();
    GGML_ASSERT(queue.തള്ളുക(ev->സന്ദേശം));
    തിരികെ പുതിയ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

ശൂന്യം rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

ശൂന്യം rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

ശൂന്യം rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->സന്ദേശം = std::make_shared<rpc_സന്ദേശം>();
    ev->സന്ദേശം->cmd = RPC_CMD_NONE;
    ev->sf = ev->സന്ദേശം->completion.get_future().share();
    GGML_ASSERT(queue.തള്ളുക(ev->സന്ദേശം));
}

ശൂന്യം rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto സന്ദേശം = std::make_shared<rpc_സന്ദേശം>();
    സന്ദേശം->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.തള്ളുക(സന്ദേശം));
    സന്ദേശം->completion.get_future().wait();
}

ശൂന്യം rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

ശൂന്യം rpc_dispatcher::busy_spin_release() {
    സ്ഥിരം unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

ശൂന്യം rpc_dispatcher::start(സ്ഥിരം std::string & endpoint) {
    std::string host;
    പൂർണ്ണസംഖ്യ port;
    എങ്കിൽ (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    എങ്കിൽ (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    എങ്കിൽ (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    എങ്കിൽ (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = സത്യം;
    thread = std::thread(rpc_dispatcher_trampoline, ഇത്);
}

ശൂന്യം rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    ആകുമ്പോൾ (running) {
        rpc_സന്ദേശം_ptr സന്ദേശം_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        സ്ഥിരം തർക്കം spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        എങ്കിൽ (spin) {
            എങ്കിൽ (!queue.try_എടുക്കുക(&സന്ദേശം_ptr)) {
                rpc_cpu_relax();
                തുടരുക;
            }
        } അല്ലെങ്കിൽ എങ്കിൽ (!queue.എടുക്കുക(&സന്ദേശം_ptr)) {
            നിർത്തുക;
        }
        എങ്കിൽ (സന്ദേശം_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(സന്ദേശം_ptr->cmd),
                     rpc_log_format_size(സന്ദേശം_ptr->നിവേശം_size).c_str(), rpc_log_format_size(സന്ദേശം_ptr->output_size).c_str());
            സ്ഥിരം തർക്കം trace = rpc_log_debug_level() >= 3;
            സ്ഥിരം auto t0 = std::chrono::steady_clock::now();
            എങ്കിൽ (സന്ദേശം_ptr->output) {
                തർക്കം status = send_rpc_cmd(sock, സന്ദേശം_ptr->cmd, സന്ദേശം_ptr->നിവേശം.get(), സന്ദേശം_ptr->നിവേശം_size, സന്ദേശം_ptr->output, സന്ദേശം_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } അല്ലെങ്കിൽ {
                തർക്കം status = send_rpc_cmd(sock, സന്ദേശം_ptr->cmd, സന്ദേശം_ptr->നിവേശം.get(), സന്ദേശം_ptr->നിവേശം_size);
                RPC_STATUS_ASSERT(status);
            }
            എങ്കിൽ (trace) {
                സ്ഥിരം std::chrono::duration<ഇരട്ട, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(സന്ദേശം_ptr->cmd), elapsed.count());
            }
        } അല്ലെങ്കിൽ {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        സന്ദേശം_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = അസത്യം;
    queue.interrupt();
    sock = nullptr;
    എങ്കിൽ (thread.joinable()) {
        thread.join();
    }
}

സ്ഥിതം std::shared_ptr<rpc_dispatcher> get_dispatcher(സ്ഥിരം std::string & endpoint) {
    സ്ഥിതം std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    സ്ഥിതം std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    എങ്കിൽ (it != dispatchers.end()) {
        എങ്കിൽ (auto dispatcher = it->second.lock()) {
            തിരികെ dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    തിരികെ dispatcher;
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_സന്ദേശം_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, വലുപ്പം(*request));
    delete ctx;
}

സ്ഥിതം ശൂന്യം * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    എങ്കിൽ (ctx->base_ptr != nullptr) {
        തിരികെ ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_സന്ദേശം_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_സന്ദേശം_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, വലുപ്പം(*request), &response, വലുപ്പം(response));
    ctx->base_ptr = reinterpret_cast<ശൂന്യം *>(response.base_ptr);
    തിരികെ ctx->base_ptr;
}

സ്ഥിതം തർക്കം ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    തിരികെ buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

സ്ഥിതം rpc_tensor serialize_tensor(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    എങ്കിൽ (!tensor) {
        memset(&result, 0, വലുപ്പം(result));
        തിരികെ result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    എങ്കിൽ (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        എങ്കിൽ (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } അല്ലെങ്കിൽ {
            result.buffer = 0;
            result.data = 0;
        }
    } അല്ലെങ്കിൽ {
        result.buffer = 0;
        result.data   = 0;
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
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, വലുപ്പം(result.name));
    result.use_എണ്ണം = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    തിരികെ result;
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    എങ്കിൽ (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_സന്ദേശം_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, വലുപ്പം(*request));
    }
    തിരികെ GGML_STATUS_SUCCESS;
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_സന്ദേശം_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, വലുപ്പം(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
സ്ഥിതം std::shared_ptr<uint8_t> serialize_set_tensor(സ്ഥിരം rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, സ്ഥിരം ശൂന്യം * data, size_t size, size_t & നിവേശം_size) {
    നിവേശം_size = വലുപ്പം(rpc_tensor) + വലുപ്പം(cache_flag) + വലുപ്പം(offset) + size;
    uint8_t * നിവേശം = പുതിയ uint8_t[നിവേശം_size]();
    uint8_t * p = നിവേശം;
    memcpy(p, &rpc_tensor, വലുപ്പം(rpc_tensor)); p += വലുപ്പം(rpc_tensor);
    memcpy(p, &cache_flag, വലുപ്പം(cache_flag)); p += വലുപ്പം(cache_flag);
    memcpy(p, &offset,     വലുപ്പം(offset));     p += വലുപ്പം(offset);
    memcpy(p, data, size);
    തിരികെ std::shared_ptr<uint8_t>(നിവേശം, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
സ്ഥിതം തർക്കം rpc_use_hash_cache(സ്ഥിരം ggml_tensor * tensor, size_t size) {
    തിരികെ size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    എങ്കിൽ (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_സന്ദേശം_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((സ്ഥിരം uint8_t*)data, size);
        rpc_സന്ദേശം_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, വലുപ്പം(*request), &response, വലുപ്പം(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        എങ്കിൽ (response.result) {
            // the server has the same data, no need to send it
            തിരികെ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t നിവേശം_size;
    auto നിവേശം = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, നിവേശം_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, നിവേശം, നിവേശം_size);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t നിവേശം_size = വലുപ്പം(rpc_tensor) + 4*വലുപ്പം(uint64_t) + size*n_copies;
    uint8_t * നിവേശം = പുതിയ uint8_t[നിവേശം_size]();
    uint8_t * dest = നിവേശം;
    memcpy(dest, &rpc_tensor, വലുപ്പം(rpc_tensor));
    dest += വലുപ്പം(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, വലുപ്പം(header));
    dest += വലുപ്പം(header);
    വേണ്ടി (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (സ്ഥിരം അക്ഷരം *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> നിവേശം_ptr(നിവേശം, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, നിവേശം_ptr, നിവേശം_size);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_സന്ദേശം_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    എങ്കിൽ (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, വലുപ്പം(*request), data, size*n_copies);
    } അല്ലെങ്കിൽ {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, വലുപ്പം(*request), packed.data(), packed.size());
        വേണ്ടി (size_t i = 0; i < n_copies; i++) {
            memcpy((അക്ഷരം *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_സന്ദേശം_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, വലുപ്പം(*request), data, size);
}

സ്ഥിതം തർക്കം ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, സ്ഥിരം ggml_tensor * src, ggml_tensor * dst) {
    എങ്കിൽ (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        എങ്കിൽ (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            തിരികെ അസത്യം;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_സന്ദേശം_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_സന്ദേശം_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, വലുപ്പം(*request), &response, വലുപ്പം(response));
        തിരികെ response.result;
    }
    തിരികെ അസത്യം;
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_സന്ദേശം_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, വലുപ്പം(*request));
}

സ്ഥിതം ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
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
    /* .reset           = */ ശൂന്യ,
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    തിരികെ buft_ctx->name.c_str();
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_സന്ദേശം_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_സന്ദേശം_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, വലുപ്പം(*request), &response, വലുപ്പം(response));
    എങ്കിൽ (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            പുതിയ ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        തിരികെ buffer;
    } അല്ലെങ്കിൽ {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        തിരികെ nullptr;
    }
}

സ്ഥിതം size_t get_alignment(സ്ഥിരം std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_സന്ദേശം_get_alignment_req>();
    request->device = device;
    rpc_സന്ദേശം_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, വലുപ്പം(*request), &response, വലുപ്പം(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    തിരികെ response.alignment;
}

സ്ഥിതം size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    തിരികെ buft_ctx->alignment;
}

സ്ഥിതം size_t get_max_size(സ്ഥിരം std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_സന്ദേശം_get_max_size_req>();
    request->device = device;
    rpc_സന്ദേശം_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, വലുപ്പം(*request), &response, വലുപ്പം(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    തിരികെ response.max_size;
}

സ്ഥിതം size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    തിരികെ buft_ctx->max_size;
}

സ്ഥിതം size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    തർക്കം rpc_get = അസത്യം;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    എങ്കിൽ (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        സ്ഥിരം size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        ഘടന alloc_size_cache_താക്കോൽ {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / വലുപ്പം(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_താക്കോൽ താക്കോൽ = {};
        താക്കോൽ.device = buft_ctx->device;
        താക്കോൽ.type = tensor->type;
        താക്കോൽ.op = tensor->op;
        memcpy(താക്കോൽ.op_params, tensor->op_params, വലുപ്പം(താക്കോൽ.op_params));
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
            താക്കോൽ.ne[i] = (uint32_t)tensor->ne[i];
            താക്കോൽ.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((സ്ഥിരം uint8_t *)&താക്കോൽ, വലുപ്പം(താക്കോൽ));
        cache_hash = fnv_hash((സ്ഥിരം uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        സ്ഥിതം std::mutex cache_mutex;
        സ്ഥിതം std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            എങ്കിൽ (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                തിരികെ std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_സന്ദേശം_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_സന്ദേശം_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, വലുപ്പം(*request), &response, വലുപ്പം(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        തിരികെ std::max<size_t>(response.alloc_size, min_size);
    }

    തിരികെ ggml_nbytes(tensor);
}

സ്ഥിതം ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ശൂന്യ,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ശൂന്യ,
    /* .is_host             = */ ശൂന്യ,
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    തിരികെ rpc_ctx->name.c_str();
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    എങ്കിൽ (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_സന്ദേശം_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((സ്ഥിരം uint8_t*)data, size);
        rpc_സന്ദേശം_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, വലുപ്പം(*request), &response, വലുപ്പം(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        എങ്കിൽ (response.result) {
            // the server has the same data, no need to send it
            തിരികെ;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t നിവേശം_size;
    auto നിവേശം = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, നിവേശം_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, നിവേശം, നിവേശം_size);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_സന്ദേശം_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, വലുപ്പം(*request), data, size);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

സ്ഥിതം ശൂന്യം add_tensor(ggml_tensor * tensor, സ്ഥിരം ggml_cgraph * cgraph, സ്ഥിരം rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    എങ്കിൽ (tensor == nullptr) {
        തിരികെ;
    }
    എങ്കിൽ (visited.find(tensor) != visited.end()) {
        തിരികെ;
    }
    visited.insert(tensor);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    സ്ഥിരം size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    എങ്കിൽ (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_എണ്ണം = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

സ്ഥിതം uint8_t * serialize_graph(uint32_t device, സ്ഥിരം ggml_cgraph * cgraph, സ്ഥിരം rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    വേണ്ടി (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*വലുപ്പം(uint32_t) + വലുപ്പം(uint64_t) + n_nodes * വലുപ്പം(uint64_t) + വലുപ്പം(uint32_t) + n_tensors * വലുപ്പം(rpc_tensor);
    uint8_t * output = പുതിയ uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, വലുപ്പം(device));
    dest += വലുപ്പം(device);
    memcpy(dest, &cgraph->uid, വലുപ്പം(cgraph->uid));
    dest += വലുപ്പം(cgraph->uid);
    memcpy(dest, &n_nodes, വലുപ്പം(n_nodes));
    dest += വലുപ്പം(n_nodes);
    വേണ്ടി (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * വലുപ്പം(uint64_t), &cgraph->nodes[i], വലുപ്പം(uint64_t));
    }
    dest += n_nodes * വലുപ്പം(uint64_t);
    memcpy(dest, &n_tensors, വലുപ്പം(n_tensors));
    dest += വലുപ്പം(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * വലുപ്പം(rpc_tensor));
    തിരികെ output;
}

ശൂന്യം rpc_dispatcher::graph_compute(uint32_t device, സ്ഥിരം ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    തർക്കം reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    എങ്കിൽ (reuse) {
        auto request = std::make_shared<rpc_സന്ദേശം_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, വലുപ്പം(*request));
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (cgraph->uid != 0) {
            എങ്കിൽ (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t നിവേശം_size = 0;
        uint8_t * നിവേശം = serialize_graph(device, cgraph, ഇത്, &നിവേശം_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(നിവേശം_size).c_str());
        std::shared_ptr<uint8_t> നിവേശം_ptr(നിവേശം, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, നിവേശം_ptr, നിവേശം_size);
    }
}

സ്ഥിതം എണ്ണൽ ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    തിരികെ GGML_STATUS_SUCCESS;
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, സ്ഥിരം ggml_tensor * tensor, ശൂന്യം * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

സ്ഥിതം ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ശൂന്യ,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ ശൂന്യ,
    /* .graph_plan_free         = */ ശൂന്യ,
    /* .graph_plan_update       = */ ശൂന്യ,
    /* .graph_plan_compute      = */ ശൂന്യ,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ ശൂന്യ,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(സ്ഥിരം അക്ഷരം * endpoint, uint32_t device) {
    സ്ഥിതം std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    സ്ഥിതം std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    എങ്കിൽ (it != buft_map.end()) {
        തിരികെ it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = പുതിയ ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = പുതിയ ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    തിരികെ buft;
}

ggml_backend_t ggml_backend_rpc_init(സ്ഥിരം അക്ഷരം * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = പുതിയ ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = പുതിയ ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    തിരികെ backend;
}

തർക്കം ggml_backend_is_rpc(ggml_backend_t backend) {
    തിരികെ backend != ശൂന്യ && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

ശൂന്യം ggml_backend_rpc_get_device_memory(സ്ഥിരം അക്ഷരം * endpoint, uint32_t device, size_t * സ്വതന്ത്രം, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_സന്ദേശം_get_device_memory_req>();
    request->device = device;
    rpc_സന്ദേശം_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, വലുപ്പം(*request), &response, വലുപ്പം(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *സ്വതന്ത്രം = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

വർഗ്ഗം rpc_server {
പൊതു:
    rpc_server(std::vector<ggml_backend_t> all_backends, സ്ഥിരം അക്ഷരം * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    ശൂന്യം hello(rpc_സന്ദേശം_hello_rsp & response);
    തർക്കം alloc_buffer(സ്ഥിരം rpc_സന്ദേശം_alloc_buffer_req & request, rpc_സന്ദേശം_alloc_buffer_rsp & response);
    തർക്കം get_alignment(സ്ഥിരം rpc_സന്ദേശം_get_alignment_req & request, rpc_സന്ദേശം_get_alignment_rsp & response);
    തർക്കം get_max_size(സ്ഥിരം rpc_സന്ദേശം_get_max_size_req & request, rpc_സന്ദേശം_get_max_size_rsp & response);
    തർക്കം buffer_get_base(സ്ഥിരം rpc_സന്ദേശം_buffer_get_base_req & request, rpc_സന്ദേശം_buffer_get_base_rsp & response);
    തർക്കം free_buffer(സ്ഥിരം rpc_സന്ദേശം_free_buffer_req & request);
    തർക്കം buffer_clear(സ്ഥിരം rpc_സന്ദേശം_buffer_clear_req & request);
    തർക്കം memset_tensor(സ്ഥിരം rpc_സന്ദേശം_memset_tensor_req & request);
    തർക്കം set_tensor(സ്ഥിരം std::vector<uint8_t> & നിവേശം);
    തർക്കം set_tensor_2d(സ്ഥിരം std::vector<uint8_t> & നിവേശം);
    തർക്കം set_tensor_hash(സ്ഥിരം rpc_സന്ദേശം_set_tensor_hash_req & request, rpc_സന്ദേശം_set_tensor_hash_rsp & response);
    തർക്കം get_tensor(സ്ഥിരം rpc_സന്ദേശം_get_tensor_req & request, std::vector<uint8_t> & response);
    തർക്കം get_tensor_2d(സ്ഥിരം rpc_സന്ദേശം_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    തർക്കം copy_tensor(സ്ഥിരം rpc_സന്ദേശം_copy_tensor_req & request, rpc_സന്ദേശം_copy_tensor_rsp & response);
    തർക്കം graph_compute(സ്ഥിരം std::vector<uint8_t> & നിവേശം);
    തർക്കം graph_recompute(സ്ഥിരം rpc_സന്ദേശം_graph_recompute_req & request);
    തർക്കം comm_init(സ്ഥിരം rpc_സന്ദേശം_comm_init_req & request, rpc_സന്ദേശം_comm_init_rsp & response);
    തർക്കം comm_allreduce(സ്ഥിരം rpc_സന്ദേശം_comm_allreduce_req & request);
    തർക്കം comm_free(സ്ഥിരം rpc_സന്ദേശം_comm_free_req & request);
    തർക്കം init_tensor(സ്ഥിരം rpc_സന്ദേശം_init_tensor_req & request);
    തർക്കം get_alloc_size(സ്ഥിരം rpc_സന്ദേശം_get_alloc_size_req & request, rpc_സന്ദേശം_get_alloc_size_rsp & response);
    തർക്കം get_device_memory(സ്ഥിരം rpc_സന്ദേശം_get_device_memory_req & request, rpc_സന്ദേശം_get_device_memory_rsp & response);

    ഘടന stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

സ്വകാര്യ:
    ശൂന്യം sync_all_backends();
    തർക്കം get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(ഘടന ggml_context * ctx, സ്ഥിരം rpc_tensor * tensor);
    ggml_tensor * create_കണ്ണി(uint64_t id,
                              ഘടന ggml_context * ctx,
                              സ്ഥിരം std::unordered_map<uint64_t, സ്ഥിരം rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, ഘടന ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    ഘടന comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    സ്ഥിരം അക്ഷരം * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

ശൂന്യം rpc_server::hello(rpc_സന്ദേശം_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

തർക്കം rpc_server::get_alloc_size(സ്ഥിരം rpc_സന്ദേശം_get_alloc_size_req & request, rpc_സന്ദേശം_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    എങ്കിൽ (dev_id >= backends.size()) {
        തിരികെ അസത്യം;
    }
    ggml_backend_buffer_type_t buft;
    ഘടന ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    എങ്കിൽ (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        തിരികെ അസത്യം;
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        എങ്കിൽ (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (ശൂന്യം*)tensor->buffer, tensor->data);
    എങ്കിൽ (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } അല്ലെങ്കിൽ {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    തിരികെ സത്യം;
}

തർക്കം rpc_server::alloc_buffer(സ്ഥിരം rpc_സന്ദേശം_alloc_buffer_req & request, rpc_സന്ദേശം_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    എങ്കിൽ (dev_id >= backends.size()) {
        തിരികെ അസത്യം;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    എങ്കിൽ (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } അല്ലെങ്കിൽ {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    തിരികെ സത്യം;
}

തർക്കം rpc_server::get_alignment(സ്ഥിരം rpc_സന്ദേശം_get_alignment_req & request, rpc_സന്ദേശം_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    എങ്കിൽ (dev_id >= backends.size()) {
        തിരികെ അസത്യം;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    തിരികെ സത്യം;
}

തർക്കം rpc_server::get_max_size(സ്ഥിരം rpc_സന്ദേശം_get_max_size_req & request, rpc_സന്ദേശം_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    എങ്കിൽ (dev_id >= backends.size()) {
        തിരികെ അസത്യം;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    തിരികെ സത്യം;
}

തർക്കം rpc_server::buffer_get_base(സ്ഥിരം rpc_സന്ദേശം_buffer_get_base_req & request, rpc_സന്ദേശം_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    എങ്കിൽ (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        തിരികെ അസത്യം;
    }
    ശൂന്യം * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    തിരികെ സത്യം;
}

തർക്കം rpc_server::free_buffer(സ്ഥിരം rpc_സന്ദേശം_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    എങ്കിൽ (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        തിരികെ അസത്യം;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    വേണ്ടി (auto & sgs : stored_graphs) {
        വേണ്ടി (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    തിരികെ സത്യം;
}

തർക്കം rpc_server::buffer_clear(സ്ഥിരം rpc_സന്ദേശം_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    എങ്കിൽ (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        തിരികെ അസത്യം;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    തിരികെ സത്യം;
}

തർക്കം rpc_server::memset_tensor(സ്ഥിരം rpc_സന്ദേശം_memset_tensor_req & request) {
    sync_all_backends();
    ഘടന ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    എങ്കിൽ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint64_t tensor_size = ggml_nbytes(tensor);
    എങ്കിൽ (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    സ്ഥിരം uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    എങ്കിൽ (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        തിരികെ അസത്യം;
    }
    സ്ഥിരം uint64_t data_offset = request.tensor.data - buffer_start;
    എങ്കിൽ (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        തിരികെ അസത്യം;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ശൂന്യം *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    തിരികെ സത്യം;
}

ggml_tensor * rpc_server::deserialize_tensor(ഘടന ggml_context * ctx, സ്ഥിരം rpc_tensor * tensor) {
    // Validate tensor type before using it
    എങ്കിൽ (tensor->type >= GGML_TYPE_എണ്ണം) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        തിരികെ nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    എങ്കിൽ (ggml_blck_size((എണ്ണൽ ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        തിരികെ nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    എങ്കിൽ (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        തിരികെ nullptr;
    }

    വേണ്ടി (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    എങ്കിൽ (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    എങ്കിൽ (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        എങ്കിൽ (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            തിരികെ nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    വേണ്ടി (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / വലുപ്പം(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<ശൂന്യം *>(tensor->data);
    ggml_set_name(result, tensor->name);
    തിരികെ result;
}


തർക്കം rpc_server::set_tensor(സ്ഥിരം std::vector<uint8_t> & നിവേശം) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    സ്ഥിരം size_t header_size = വലുപ്പം(rpc_tensor) + വലുപ്പം(cache_flag) + വലുപ്പം(offset);
    എങ്കിൽ (നിവേശം.size() < header_size) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം rpc_tensor * in_tensor = (സ്ഥിരം rpc_tensor *)നിവേശം.data();
    memcpy(&cache_flag, നിവേശം.data() + വലുപ്പം(rpc_tensor), വലുപ്പം(cache_flag));
    memcpy(&offset,     നിവേശം.data() + വലുപ്പം(rpc_tensor) + വലുപ്പം(cache_flag), വലുപ്പം(offset));
    സ്ഥിരം size_t size = നിവേശം.size() - header_size;

    ഘടന ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    എങ്കിൽ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        തിരികെ അസത്യം;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ശൂന്യം*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        സ്ഥിരം size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        സ്ഥിരം size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        എങ്കിൽ (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            തിരികെ അസത്യം;
        }
    }

    സ്ഥിരം ശൂന്യം * data = നിവേശം.data() + header_size;
    എങ്കിൽ (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((സ്ഥിരം uint8_t*)data, size);
        അക്ഷരം hash_str[17];
        snprintf(hash_str, വലുപ്പം(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.ലിഖ((സ്ഥിരം അക്ഷരം *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    തിരികെ സത്യം;
}

തർക്കം rpc_server::set_tensor_2d(സ്ഥിരം std::vector<uint8_t> & നിവേശം) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    എങ്കിൽ (നിവേശം.size() < വലുപ്പം(rpc_tensor) + 4*വലുപ്പം(uint64_t)) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം rpc_tensor * in_tensor = (സ്ഥിരം rpc_tensor *)നിവേശം.data();
    uint64_t header[4];
    memcpy(header, നിവേശം.data() + വലുപ്പം(rpc_tensor), വലുപ്പം(header));
    സ്ഥിരം uint64_t offset   = header[0];
    സ്ഥിരം uint64_t size     = header[1];
    സ്ഥിരം uint64_t n_copies = header[2];
    സ്ഥിരം uint64_t stride   = header[3];

    സ്ഥിരം uint64_t data_size = നിവേശം.size() - വലുപ്പം(rpc_tensor) - 4*വലുപ്പം(uint64_t);
    എങ്കിൽ (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        തിരികെ അസത്യം;
    }

    ഘടന ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    എങ്കിൽ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        തിരികെ അസത്യം;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ശൂന്യം*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        എങ്കിൽ (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം uint64_t span = (n_copies - 1)*stride + size;
        സ്ഥിരം uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        സ്ഥിരം uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        എങ്കിൽ (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            തിരികെ അസത്യം;
        }
    }

    സ്ഥിരം ശൂന്യം * data = നിവേശം.data() + വലുപ്പം(rpc_tensor) + 4*വലുപ്പം(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    തിരികെ സത്യം;
}

തർക്കം rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    എങ്കിൽ (!cache_dir) {
        തിരികെ അസത്യം;
    }
    അക്ഷരം hash_str[17];
    snprintf(hash_str, വലുപ്പം(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    എങ്കിൽ (!fs::exists(cache_file, ec)) {
        തിരികെ അസത്യം;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.പഠ((അക്ഷരം *)data.data(), size);
    തിരികെ സത്യം;
}

തർക്കം rpc_server::set_tensor_hash(സ്ഥിരം rpc_സന്ദേശം_set_tensor_hash_req & request, rpc_സന്ദേശം_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    ഘടന ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    എങ്കിൽ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        തിരികെ അസത്യം;
    }

    std::vector<uint8_t> cached_file;
    എങ്കിൽ (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        തിരികെ സത്യം;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ശൂന്യം*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        സ്ഥിരം size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        സ്ഥിരം size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        എങ്കിൽ (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            തിരികെ അസത്യം;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    തിരികെ സത്യം;
}

തർക്കം rpc_server::init_tensor(സ്ഥിരം rpc_സന്ദേശം_init_tensor_req & request) {
    sync_all_backends();
    ഘടന ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    എങ്കിൽ (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        തിരികെ അസത്യം;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (ശൂന്യം*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    എങ്കിൽ (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    എങ്കിൽ (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

തർക്കം rpc_server::get_tensor(സ്ഥിരം rpc_സന്ദേശം_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ഘടന ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    എങ്കിൽ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        തിരികെ അസത്യം;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (ശൂന്യം*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        സ്ഥിരം size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        സ്ഥിരം size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        എങ്കിൽ (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                തിരികെ അസത്യം;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    തിരികെ സത്യം;
}

തർക്കം rpc_server::get_tensor_2d(സ്ഥിരം rpc_സന്ദേശം_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ഘടന ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    എങ്കിൽ (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        തിരികെ അസത്യം;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (ശൂന്യം*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        എങ്കിൽ (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        സ്ഥിരം uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        സ്ഥിരം uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        എങ്കിൽ (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            തിരികെ അസത്യം;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    തിരികെ സത്യം;
}

തർക്കം rpc_server::copy_tensor(സ്ഥിരം rpc_സന്ദേശം_copy_tensor_req & request, rpc_സന്ദേശം_copy_tensor_rsp & response) {
    sync_all_backends();
    ഘടന ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    എങ്കിൽ (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        തിരികെ അസത്യം;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    എങ്കിൽ (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        തിരികെ അസത്യം;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    തിരികെ സത്യം;
}

ggml_tensor * rpc_server::create_കണ്ണി(uint64_t id,
                                      ഘടന ggml_context * ctx,
                                      സ്ഥിരം std::unordered_map<uint64_t, സ്ഥിരം rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, ഘടന ggml_tensor*> & tensor_map) {
    എങ്കിൽ (tensor_map.find(id) != tensor_map.end()) {
        തിരികെ tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    എങ്കിൽ (it_ptr == tensor_ptrs.end()) {
        തിരികെ nullptr;
    }
    സ്ഥിരം rpc_tensor * tensor = it_ptr->second;

    ഘടന ggml_tensor * result = deserialize_tensor(ctx, tensor);
    എങ്കിൽ (result == nullptr) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        തിരികെ nullptr;
    }
    tensor_map[id] = result;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        എങ്കിൽ (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } അല്ലെങ്കിൽ {
            result->src[i] = create_കണ്ണി(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            എങ്കിൽ (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                തിരികെ nullptr;
            }
        }
    }

    // Handle view_src similarly
    എങ്കിൽ (tensor->view_src == 0) {
        result->view_src = nullptr;
    } അല്ലെങ്കിൽ {
        result->view_src = create_കണ്ണി(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        എങ്കിൽ (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            തിരികെ nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    തിരികെ result;
}

തർക്കം rpc_server::graph_compute(സ്ഥിരം std::vector<uint8_t> & നിവേശം) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    എങ്കിൽ (നിവേശം.size() < 2*വലുപ്പം(uint32_t) + വലുപ്പം(uint64_t)) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം uint8_t * src = നിവേശം.data();
    uint32_t device;
    memcpy(&device, src, വലുപ്പം(device));
    src += വലുപ്പം(device);
    എങ്കിൽ (device >= backends.size()) {
        തിരികെ അസത്യം;
    }
    uint64_t uid;
    memcpy(&uid, src, വലുപ്പം(uid));
    src += വലുപ്പം(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, വലുപ്പം(n_nodes));
    src += വലുപ്പം(n_nodes);
    എങ്കിൽ (നിവേശം.size() < 2*വലുപ്പം(uint32_t) + വലുപ്പം(uint64_t) + n_nodes*വലുപ്പം(uint64_t) + വലുപ്പം(uint32_t)) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം uint64_t * nodes = (സ്ഥിരം uint64_t *)src;
    src += n_nodes*വലുപ്പം(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, വലുപ്പം(n_tensors));
    src += വലുപ്പം(n_tensors);
    എങ്കിൽ (നിവേശം.size() < 2*വലുപ്പം(uint32_t) + വലുപ്പം(uint64_t) + n_nodes*വലുപ്പം(uint64_t) + വലുപ്പം(uint32_t) + n_tensors*വലുപ്പം(rpc_tensor)) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം rpc_tensor * tensors = (സ്ഥിരം rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    എങ്കിൽ (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, അസത്യം);
    എങ്കിൽ (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    ഘടന ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ഘടന ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, അസത്യം);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, സ്ഥിരം rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    വേണ്ടി (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    സ്ഥിരം auto t_deserialize = std::chrono::steady_clock::now();
    വേണ്ടി (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], വലുപ്പം(id));
        graph->nodes[i] = create_കണ്ണി(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        എങ്കിൽ (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            സ്ഥിരം size_t hash_pos = ggml_hash_തിരുകുക(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_എണ്ണം;
        }
    }
    സ്ഥിരം std::chrono::duration<ഇരട്ട, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    തിരികെ സത്യം;
}

തർക്കം rpc_server::graph_recompute(സ്ഥിരം rpc_സന്ദേശം_graph_recompute_req & request) {
    uint32_t device = request.device;
    എങ്കിൽ (device >= backends.size()) {
        തിരികെ അസത്യം;
    }
    auto it = stored_graphs[device].find(request.uid);
    എങ്കിൽ (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        തിരികെ അസത്യം;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    തിരികെ സത്യം;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
ശൂന്യം rpc_server::sync_all_backends() {
    വേണ്ടി (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
തർക്കം rpc_server::comm_init(സ്ഥിരം rpc_സന്ദേശം_comm_init_req & request, rpc_സന്ദേശം_comm_init_rsp & response) {
    response.ok = 0;
    എങ്കിൽ (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        തിരികെ സത്യം;
    }
    comm_state & state = comm_states[request.device];
    എങ്കിൽ (state.peer != nullptr) {
        response.ok = 1;
        തിരികെ സത്യം;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    എങ്കിൽ (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        എങ്കിൽ (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            തിരികെ സത്യം;
        }
        state.peer = srv->accept();
        എങ്കിൽ (state.peer == nullptr) {
            തിരികെ സത്യം;
        }
        എങ്കിൽ (!state.peer->recv_data(remote_caps, വലുപ്പം(remote_caps))) {
            state.peer = nullptr;
            തിരികെ സത്യം;
        }
        state.peer->get_caps(local_caps);
        എങ്കിൽ (!state.peer->send_data(local_caps, വലുപ്പം(local_caps))) {
            state.peer = nullptr;
            തിരികെ സത്യം;
        }
        state.peer->update_caps(remote_caps);
    } അല്ലെങ്കിൽ {
        സ്ഥിരം std::string host(request.host, strnlen(request.host, വലുപ്പം(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            എങ്കിൽ (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        എങ്കിൽ (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            തിരികെ സത്യം;
        }
        state.peer->get_caps(local_caps);
        എങ്കിൽ (!state.peer->send_data(local_caps, വലുപ്പം(local_caps)) ||
            !state.peer->recv_data(remote_caps, വലുപ്പം(remote_caps))) {
            state.peer = nullptr;
            തിരികെ സത്യം;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    തിരികെ സത്യം;
}

തർക്കം rpc_server::comm_allreduce(സ്ഥിരം rpc_സന്ദേശം_comm_allreduce_req & request) {
    എങ്കിൽ (request.device >= backends.size()) {
        തിരികെ അസത്യം;
    }
    comm_state & state = comm_states[request.device];
    എങ്കിൽ (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        തിരികെ അസത്യം;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, അസത്യം);
    ഘടന ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ശൂന്യ,
        /*.no_alloc   =*/ സത്യം,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    എങ്കിൽ (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        തിരികെ അസത്യം;
    }
    സ്ഥിരം size_t  nbytes = ggml_nbytes(t_dst);
    സ്ഥിരം int64_t ne     = ggml_nelements(t_dst);
    എങ്കിൽ (nbytes == 0) {
        തിരികെ സത്യം;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    സ്ഥിരം തർക്കം   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    സ്ഥിരം size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    സ്ഥിരം size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    എങ്കിൽ (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    അക്ഷരം * scratch_base = (അക്ഷരം *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        തിരികെ t;
    };
    auto new_cpy_കണ്ണി = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        തിരികെ t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, അസത്യം);
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
    എങ്കിൽ (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_കണ്ണി(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } അല്ലെങ്കിൽ {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    എങ്കിൽ (state.rank == 0) {
        എങ്കിൽ (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            തിരികെ അസത്യം;
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            തിരികെ അസത്യം;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    എങ്കിൽ (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_കണ്ണി(t_wire_recv, t_peer);
    } അല്ലെങ്കിൽ {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    എങ്കിൽ (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } അല്ലെങ്കിൽ {
        compute_nodes(t_red, nullptr);
    }
    തിരികെ സത്യം;
}

തർക്കം rpc_server::comm_free(സ്ഥിരം rpc_സന്ദേശം_comm_free_req & request) {
    എങ്കിൽ (request.device >= backends.size()) {
        തിരികെ അസത്യം;
    }
    comm_states[request.device] = comm_state();
    തിരികെ സത്യം;
}

തർക്കം rpc_server::get_device_memory(സ്ഥിരം rpc_സന്ദേശം_get_device_memory_req & request, rpc_സന്ദേശം_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    എങ്കിൽ (dev_id >= backends.size()) {
        തിരികെ അസത്യം;
    }
    size_t സ്വതന്ത്രം, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &സ്വതന്ത്രം, &total);
    response.free_mem = സ്വതന്ത്രം;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    തിരികെ സത്യം;
}

rpc_server::~rpc_server() {
    വേണ്ടി (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

സ്ഥിതം ശൂന്യം rpc_serve_client(സ്ഥിരം std::vector<ggml_backend_t> & backends, സ്ഥിരം അക്ഷരം * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    എങ്കിൽ (!sock->recv_data(&cmd, 1)) {
        തിരികെ;
    }
    എങ്കിൽ (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        തിരികെ;
    }

    // Read input_size and validate protocol version
    uint64_t hello_നിവേശം_size;
    എങ്കിൽ (!sock->recv_data(&hello_നിവേശം_size, വലുപ്പം(hello_നിവേശം_size))) {
        തിരികെ;
    }

    എങ്കിൽ (hello_നിവേശം_size != വലുപ്പം(rpc_സന്ദേശം_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_നിവേശം_size, വലുപ്പം(rpc_സന്ദേശം_hello_req), RPC_PROTO_MAJOR_VERSION);
        തിരികെ;
    }

    rpc_സന്ദേശം_hello_req req = {};
    എങ്കിൽ (!sock->recv_data(&req, വലുപ്പം(req))) {
        തിരികെ;
    }

    rpc_സന്ദേശം_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    എങ്കിൽ (!send_സന്ദേശം(sock, &rsp, വലുപ്പം(rsp))) {
        തിരികെ;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    ആകുമ്പോൾ (സത്യം) {
        എങ്കിൽ (!sock->recv_data(&cmd, 1)) {
            നിർത്തുക;
        }
        എങ്കിൽ (cmd >= RPC_CMD_എണ്ണം) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            നിർത്തുക;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((എണ്ണൽ rpc_cmd)cmd));
        തിരഞ്ഞെടുപ്പ് (cmd) {
            അവസ്ഥ RPC_CMD_HELLO: {
                // HELLO command is handled above
                തിരികെ;
            }
            അവസ്ഥ RPC_CMD_DEVICE_എണ്ണം: {
                എങ്കിൽ (!recv_സന്ദേശം(sock, nullptr, 0)) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_device_എണ്ണം_rsp response;
                response.device_എണ്ണം = backends.size();
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_ALLOC_BUFFER: {
                rpc_സന്ദേശം_alloc_buffer_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_alloc_buffer_rsp response;
                എങ്കിൽ (!server.alloc_buffer(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_GET_ALLOC_SIZE: {
                rpc_സന്ദേശം_get_alloc_size_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_get_alloc_size_rsp response;
                എങ്കിൽ (!server.get_alloc_size(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_GET_ALIGNMENT: {
                rpc_സന്ദേശം_get_alignment_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_get_alignment_rsp response;
                എങ്കിൽ (!server.get_alignment(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_GET_MAX_SIZE: {
                rpc_സന്ദേശം_get_max_size_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_get_max_size_rsp response;
                എങ്കിൽ (!server.get_max_size(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_BUFFER_GET_BASE: {
                rpc_സന്ദേശം_buffer_get_base_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_buffer_get_base_rsp response;
                എങ്കിൽ (!server.buffer_get_base(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_FREE_BUFFER: {
                rpc_സന്ദേശം_free_buffer_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.free_buffer(request)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_BUFFER_CLEAR: {
                rpc_സന്ദേശം_buffer_clear_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.buffer_clear(request)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_MEMSET_TENSOR: {
                rpc_സന്ദേശം_memset_tensor_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.memset_tensor(request)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> നിവേശം;
                എങ്കിൽ (!recv_സന്ദേശം(sock, നിവേശം)) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.set_tensor(നിവേശം)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> നിവേശം;
                എങ്കിൽ (!recv_സന്ദേശം(sock, നിവേശം)) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.set_tensor_2d(നിവേശം)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_GET_TENSOR_2D: {
                rpc_സന്ദേശം_get_tensor_2d_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                std::vector<uint8_t> response;
                എങ്കിൽ (!server.get_tensor_2d(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, response.data(), response.size())) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_SET_TENSOR_HASH: {
                rpc_സന്ദേശം_set_tensor_hash_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_set_tensor_hash_rsp response;
                എങ്കിൽ (!server.set_tensor_hash(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_INIT_TENSOR: {
                rpc_സന്ദേശം_init_tensor_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request,വലുപ്പം(request))) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.init_tensor(request)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_GET_TENSOR: {
                rpc_സന്ദേശം_get_tensor_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                std::vector<uint8_t> response;
                എങ്കിൽ (!server.get_tensor(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, response.data(), response.size())) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_COPY_TENSOR: {
                rpc_സന്ദേശം_copy_tensor_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_copy_tensor_rsp response;
                എങ്കിൽ (!server.copy_tensor(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> നിവേശം;
                എങ്കിൽ (!recv_സന്ദേശം(sock, നിവേശം)) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.graph_compute(നിവേശം)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_സന്ദേശം_graph_recompute_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.graph_recompute(request)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_COMM_INIT: {
                rpc_സന്ദേശം_comm_init_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_comm_init_rsp response;
                എങ്കിൽ (!server.comm_init(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_COMM_ALLREDUCE: {
                rpc_സന്ദേശം_comm_allreduce_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.comm_allreduce(request)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_COMM_FREE: {
                rpc_സന്ദേശം_comm_free_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                എങ്കിൽ (!server.comm_free(request)) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            അവസ്ഥ RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_സന്ദേശം_get_device_memory_req request;
                എങ്കിൽ (!recv_സന്ദേശം(sock, &request, വലുപ്പം(request))) {
                    തിരികെ;
                }
                rpc_സന്ദേശം_get_device_memory_rsp response;
                എങ്കിൽ (!server.get_device_memory(request, response)) {
                    തിരികെ;
                }
                എങ്കിൽ (!send_സന്ദേശം(sock, &response, വലുപ്പം(response))) {
                    തിരികെ;
                }
                നിർത്തുക;
            }
            സ്ഥിരസ്ഥിതി: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((എണ്ണൽ rpc_cmd)cmd));
                തിരികെ;
            }
        }
    }
}

ശൂന്യം ggml_backend_rpc_start_server(സ്ഥിരം അക്ഷരം * endpoint, സ്ഥിരം അക്ഷരം * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    എങ്കിൽ (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        തിരികെ;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    വേണ്ടി (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t സ്വതന്ത്രം, total;
        ggml_backend_dev_memory(dev, &സ്വതന്ത്രം, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, സ്വതന്ത്രം / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        എങ്കിൽ (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            തിരികെ;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        എങ്കിൽ (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            എങ്കിൽ (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    പൂർണ്ണസംഖ്യ port;
    എങ്കിൽ (!parse_endpoint(endpoint, host, port)) {
        തിരികെ;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#അല്ലെങ്കിൽ
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    എങ്കിൽ (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        തിരികെ;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    എങ്കിൽ (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        തിരികെ;
    }
    ആകുമ്പോൾ (സത്യം) {
        auto client_socket = server_socket->accept();
        എങ്കിൽ (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            തിരികെ;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    വേണ്ടി (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    തിരികെ ctx->name.c_str();
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    തിരികെ ctx->description.c_str();
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, സ്വതന്ത്രം, total);
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    തിരികെ GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, ഘടന ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ സത്യം,
        /* .host_buffer           = */ അസത്യം,
        /* .buffer_from_host_ptr  = */ അസത്യം,
        /* .events                = */ സത്യം,
        /* .mmap_support          = */ സത്യം,
    };
}

സ്ഥിതം ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    തിരികെ ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    തിരികെ ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

സ്ഥിതം തർക്കം ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ഘടന ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    എങ്കിൽ (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        തിരികെ അസത്യം;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    തിരികെ buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

സ്ഥിതം ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    തിരികെ dispatcher->event_new(dev);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

സ്ഥിതം ശൂന്യം ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ ശൂന്യ,
    /* .buffer_from_host_ptr = */ ശൂന്യ,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ ശൂന്യ,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

ഘടന ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    തിരികെ ctx ? ctx->name.c_str() : "RPC";
}

സ്ഥിതം size_t ggml_backend_rpc_reg_get_device_എണ്ണം(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    തിരികെ ctx ? ctx->devices.size() : 0;
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    എങ്കിൽ (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(index < ctx->devices.size());
        തിരികെ ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
ഘടന ggml_backend_rpc_comm_context {
    ഘടന rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

സ്ഥിതം ശൂന്യം ggml_backend_rpc_comm_free(ശൂന്യം * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    എങ്കിൽ (comm_ctx == nullptr) {
        തിരികെ;
    }
    വേണ്ടി (സ്ഥിരം auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_സന്ദേശം_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, വലുപ്പം(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

സ്ഥിതം ശൂന്യം * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    എങ്കിൽ (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        തിരികെ nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        എങ്കിൽ (!ggml_backend_is_rpc(backends[i])) {
            തിരികെ nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        വേണ്ടി (സ്ഥിരം auto & rank : ranks) {
            എങ്കിൽ (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                തിരികെ nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    പൂർണ്ണസംഖ്യ port0;
    എങ്കിൽ (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        തിരികെ nullptr;
    }
    സ്ഥിരം uint32_t comm_port = (uint32_t) port0 + 1000;
    എങ്കിൽ (host0.size() >= 64) {
        തിരികെ nullptr;
    }

    വേണ്ടി (സ്ഥിരം auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_സന്ദേശം_comm_init_rsp> responses(n_backends);
    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_സന്ദേശം_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        എങ്കിൽ (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, വലുപ്പം(*request), &responses[i], വലുപ്പം(responses[i]));
    }
    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    തർക്കം ok = സത്യം;
    വേണ്ടി (size_t i = 0; i < n_backends; i++) {
        എങ്കിൽ (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = അസത്യം;
        }
    }
    എങ്കിൽ (!ok) {
        വേണ്ടി (സ്ഥിരം auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        തിരികെ nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    തിരികെ പുതിയ ggml_backend_rpc_comm_context{std::move(ranks)};
}

സ്ഥിതം തർക്കം ggml_backend_rpc_comm_allreduce_tensor(ശൂന്യം * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    എങ്കിൽ (comm_ctx == nullptr) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം size_t n_ranks = comm_ctx->ranks.size();
    സ്ഥിരം int64_t ne = ggml_nelements(tensors[0]);
    എങ്കിൽ (ne == 0) {
        തിരികെ സത്യം;
    }
    വേണ്ടി (size_t i = 0; i < n_ranks; i++) {
        എങ്കിൽ (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            തിരികെ അസത്യം;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        എങ്കിൽ ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തിരികെ അസത്യം;
        }
    }
    വേണ്ടി (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_സന്ദേശം_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, വലുപ്പം(*request));
    }
    തിരികെ സത്യം;
}

സ്ഥിതം ശൂന്യം * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം * name) {
    എങ്കിൽ (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_rpc_add_server;
    }
    എങ്കിൽ (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_rpc_start_server;
    }
    എങ്കിൽ (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_rpc_comm_init;
    }
    എങ്കിൽ (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_rpc_comm_free;
    }
    എങ്കിൽ (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    തിരികെ ശൂന്യ;

    GGML_UNUSED(reg);
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_എണ്ണം,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(ശൂന്യം) {
    സ്ഥിതം ഘടന ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ ശൂന്യ,
    };

    തിരികെ &ggml_backend_rpc_reg;
}

സ്ഥിതം uint32_t ggml_backend_rpc_get_device_എണ്ണം(സ്ഥിരം അക്ഷരം * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_സന്ദേശം_device_എണ്ണം_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_എണ്ണം, nullptr, 0, &response, വലുപ്പം(response));
    തിരികെ response.device_എണ്ണം;
}

സ്ഥിതം സ്ഥിരം ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_എണ്ണം,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(സ്ഥിരം അക്ഷരം * endpoint) {
    സ്ഥിതം std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    സ്ഥിതം std::mutex mutex;
    സ്ഥിതം uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    എങ്കിൽ (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        തിരികെ reg_map[endpoint];
    }
    uint32_t dev_എണ്ണം = ggml_backend_rpc_get_device_എണ്ണം(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_എണ്ണം);
    എങ്കിൽ (dev_എണ്ണം == 0) {
        തിരികെ nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = പുതിയ ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    വേണ്ടി (uint32_t ind = 0; ind < dev_എണ്ണം; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = പുതിയ ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = പുതിയ ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = പുതിയ ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    തിരികെ reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

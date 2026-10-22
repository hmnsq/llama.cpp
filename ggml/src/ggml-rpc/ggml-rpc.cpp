#અન્તર્ભાવ "ggml-rpc.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-cpp.h"
#અન્તર્ભાવ "transport.h"
#અન્તર્ભાવ "log.h"

#અન્તર્ભાવ <array>
#અન્તર્ભાવ <chrono>
#અન્તર્ભાવ <cinttypes>
#અન્તર્ભાવ <optional>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <thread>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <queue>
#અન્તર્ભાવ <condition_variable>
#અન્તર્ભાવ <future>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <unordered_map>
#અન્તર્ભાવ <unordered_set>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <filesystem>
#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <atomic>

નામાકાશ fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) જો (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(ધકેલો, 1)
// ggml_tensor is serialized into rpc_tensor
સંરચના rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / કદ(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    અક્ષર name[GGML_MAX_NAME];

    int32_t use_ગણતરી;
};

static_assert(કદ(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
ગણના rpc_cmd {
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
    RPC_CMD_DEVICE_ગણતરી,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_ગણતરી,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

સ્થિર અચલ અક્ષર * rpc_cmd_name(ગણના rpc_cmd cmd) {
    પસંદગી (cmd) {
        સ્થિતિ RPC_CMD_ALLOC_BUFFER:      પાછા "ALLOC_BUFFER";
        સ્થિતિ RPC_CMD_GET_ALIGNMENT:     પાછા "GET_ALIGNMENT";
        સ્થિતિ RPC_CMD_GET_MAX_SIZE:      પાછા "GET_MAX_SIZE";
        સ્થિતિ RPC_CMD_BUFFER_GET_BASE:   પાછા "BUFFER_GET_BASE";
        સ્થિતિ RPC_CMD_FREE_BUFFER:       પાછા "FREE_BUFFER";
        સ્થિતિ RPC_CMD_BUFFER_CLEAR:      પાછા "BUFFER_CLEAR";
        સ્થિતિ RPC_CMD_SET_TENSOR:        પાછા "SET_TENSOR";
        સ્થિતિ RPC_CMD_SET_TENSOR_HASH:   પાછા "SET_TENSOR_HASH";
        સ્થિતિ RPC_CMD_GET_TENSOR:        પાછા "GET_TENSOR";
        સ્થિતિ RPC_CMD_COPY_TENSOR:       પાછા "COPY_TENSOR";
        સ્થિતિ RPC_CMD_GRAPH_COMPUTE:     પાછા "GRAPH_COMPUTE";
        સ્થિતિ RPC_CMD_GET_DEVICE_MEMORY: પાછા "GET_DEVICE_MEMORY";
        સ્થિતિ RPC_CMD_INIT_TENSOR:       પાછા "INIT_TENSOR";
        સ્થિતિ RPC_CMD_GET_ALLOC_SIZE:    પાછા "GET_ALLOC_SIZE";
        સ્થિતિ RPC_CMD_HELLO:             પાછા "HELLO";
        સ્થિતિ RPC_CMD_DEVICE_ગણતરી:      પાછા "DEVICE_COUNT";
        સ્થિતિ RPC_CMD_GRAPH_RECOMPUTE:   પાછા "GRAPH_RECOMPUTE";
        સ્થિતિ RPC_CMD_MEMSET_TENSOR:     પાછા "MEMSET_TENSOR";
        સ્થિતિ RPC_CMD_SET_TENSOR_2D:     પાછા "SET_TENSOR_2D";
        સ્થિતિ RPC_CMD_GET_TENSOR_2D:     પાછા "GET_TENSOR_2D";
        સ્થિતિ RPC_CMD_COMM_INIT:         પાછા "COMM_INIT";
        સ્થિતિ RPC_CMD_COMM_ALLREDUCE:    પાછા "COMM_ALLREDUCE";
        સ્થિતિ RPC_CMD_COMM_FREE:         પાછા "COMM_FREE";
        સ્થિતિ RPC_CMD_NONE:              પાછા "NONE";
        મૂળભૂત:                        પાછા "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
અચલ size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
અચલ size_t GRAPH_CACHE_MAX = 1024;

સંરચના rpc_સંદેશ_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

સંરચના rpc_સંદેશ_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

સંરચના rpc_સંદેશ_device_ગણતરી_rsp {
    uint32_t device_ગણતરી;
};

સંરચના rpc_સંદેશ_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

સંરચના rpc_સંદેશ_get_alloc_size_rsp {
    uint64_t alloc_size;
};

સંરચના rpc_સંદેશ_init_tensor_req {
    rpc_tensor tensor;
};

સંરચના rpc_સંદેશ_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

સંરચના rpc_સંદેશ_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

સંરચના rpc_સંદેશ_get_alignment_req {
    uint32_t device;
};

સંરચના rpc_સંદેશ_get_alignment_rsp {
    uint64_t alignment;
};

સંરચના rpc_સંદેશ_get_max_size_req {
    uint32_t device;
};

સંરચના rpc_સંદેશ_get_max_size_rsp {
    uint64_t max_size;
};

સંરચના rpc_સંદેશ_buffer_get_base_req {
    uint64_t remote_ptr;
};

સંરચના rpc_સંદેશ_buffer_get_base_rsp {
    uint64_t base_ptr;
};

સંરચના rpc_સંદેશ_free_buffer_req {
    uint64_t remote_ptr;
};

સંરચના rpc_સંદેશ_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

સંરચના rpc_સંદેશ_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

સંરચના rpc_સંદેશ_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

સંરચના rpc_સંદેશ_set_tensor_hash_rsp {
    uint8_t result;
};

સંરચના rpc_સંદેશ_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

સંરચના rpc_સંદેશ_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

સંરચના rpc_સંદેશ_copy_tensor_rsp {
    uint8_t result;
};

સંરચના rpc_સંદેશ_get_device_memory_req {
    uint32_t device;
};

સંરચના rpc_સંદેશ_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

સંરચના rpc_સંદેશ_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

સંરચના rpc_સંદેશ_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

સંરચના rpc_સંદેશ_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    અક્ષર     host[64];  // rank > 0: rank 0's host
};

સંરચના rpc_સંદેશ_comm_init_rsp {
    uint8_t ok;
};

સંરચના rpc_સંદેશ_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

સંરચના rpc_સંદેશ_comm_free_req {
    uint32_t device;
};

#pragma pack(કાઢો)

// RPC data structures

સ્થિર ggml_guid_t ggml_backend_rpc_guid() {
    સ્થિર ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    પાછા &guid;
}

સંરચના ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

સંરચના ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

વર્ગ rpc_dispatcher;
સંરચના ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

સંરચના ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    શૂન્ય                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
સ્થિર uint64_t fnv_hash(અચલ uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    અચલ uint64_t fnv_અવિભાજ્ય = 0x100000001b3ULL;

    માટે (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_અવિભાજ્ય;
    }
    પાછા hash;
}

સ્થિર તાર્કિક send_સંદેશ(socket_ptr sock, અચલ શૂન્ય * સંદેશ, size_t સંદેશ_size) {
    જો (!sock->send_data(&સંદેશ_size, કદ(સંદેશ_size))) {
        પાછા ખોટું;
    }
    જો (!sock->send_data(સંદેશ, સંદેશ_size)) {
        પાછા ખોટું;
    }
    પાછા sock->flush();
}

સ્થિર તાર્કિક recv_સંદેશ(socket_ptr sock, શૂન્ય * સંદેશ, size_t સંદેશ_size) {
    uint64_t size;
    જો (!sock->recv_data(&size, કદ(size))) {
        પાછા ખોટું;
    }
    જો (size != સંદેશ_size) {
        પાછા ખોટું;
    }
    પાછા sock->recv_data(સંદેશ, સંદેશ_size);
}

સ્થિર તાર્કિક recv_સંદેશ(socket_ptr sock, std::vector<uint8_t> & આગત) {
    uint64_t size;
    જો (!sock->recv_data(&size, કદ(size))) {
        પાછા ખોટું;
    }
    પ્રયાસ {
        આગત.resize(size);
    } પકડો (અચલ std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        પાછા ખોટું;
    }
    પાછા sock->recv_data(આગત.data(), size);
}

સ્થિર તાર્કિક parse_endpoint(અચલ std::string & endpoint, std::string & host, પૂર્ણાંક & port) {
    size_t pos = endpoint.find(':');
    જો (pos == std::string::npos) {
        પાછા ખોટું;
    }
    host = endpoint.substr(0, pos);
    પ્રયાસ {
        port = std::stoi(endpoint.substr(pos + 1));
    } પકડો (...) {
        પાછા ખોટું;
    }
    પાછા સાચું;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
સ્થિર તાર્કિક send_rpc_cmd(socket_ptr sock, ગણના rpc_cmd cmd, અચલ શૂન્ય * આગત, size_t આગત_size) {
    uint8_t cmd_byte = cmd;
    જો (!sock->send_data(&cmd_byte, કદ(cmd_byte))) {
        પાછા ખોટું;
    }
    જો (!sock->send_data(&આગત_size, કદ(આગત_size))) {
        પાછા ખોટું;
    }
    જો (!sock->send_data(આગત, આગત_size)) {
        પાછા ખોટું;
    }
    પાછા sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
સ્થિર તાર્કિક send_rpc_cmd(socket_ptr sock, ગણના rpc_cmd cmd, અચલ શૂન્ય * આગત, size_t આગત_size, શૂન્ય * output, size_t output_size) {
    જો (!send_rpc_cmd(sock, cmd, આગત, આગત_size)) {
        પાછા ખોટું;
    }
    uint64_t out_size;
    જો (!sock->recv_data(&out_size, કદ(out_size))) {
        પાછા ખોટું;
    }
    જો (out_size != output_size) {
        પાછા ખોટું;
    }
    જો (!sock->recv_data(output, output_size)) {
        પાછા ખોટું;
    }
    પાછા સાચું;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
સ્થિર constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

સ્થિર અંતઃરેખ શૂન્ય rpc_cpu_relax() {
#જો defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ અસ્થિર("yield" ::: "memory");
#નહીંતર
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
સ્થિર તાર્કિક negotiate_hello(અચલ std::shared_ptr<socket_t> & sock) {
    rpc_સંદેશ_hello_req request = {};
    rpc_સંદેશ_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    તાર્કિક status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, કદ(request), &response, કદ(response));
    RPC_STATUS_ASSERT(status);

    જો (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        પાછા ખોટું;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    પાછા સાચું;
}

template <typename T>
વર્ગ message_queue {
જાહેર:
    message_queue() {}

    તાર્કિક ધકેલો(અચલ T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        જો (interrupted) {
            પાછા ખોટું;
        }
        queue.ધકેલો(value);
        cvar.notify_all();
        પાછા સાચું;
    }

    તાર્કિક કાઢો(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [આ] { પાછા !queue.empty() || interrupted; });
        જો (interrupted) {
            પાછા ખોટું;
        }
        *out = queue.front();
        queue.કાઢો();
        પાછા સાચું;
    }

    તાર્કિક try_કાઢો(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        જો (interrupted || queue.empty()) {
            પાછા ખોટું;
        }
        *out = queue.front();
        queue.કાઢો();
        પાછા સાચું;
    }

    શૂન્ય interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = સાચું;
        lock.unlock();
        cvar.notify_all();
    }

ખાનગી:
    તાર્કિક interrupted = ખોટું;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

વર્ગ rpc_dispatcher {
જાહેર:
    rpc_dispatcher() {
    }

    શૂન્ય send(ગણના rpc_cmd cmd, std::shared_ptr<અચલ શૂન્ય> આગત, size_t આગત_size);
    શૂન્ય send(ગણના rpc_cmd cmd, std::shared_ptr<અચલ શૂન્ય> આગત, size_t આગત_size, શૂન્ય * output, size_t output_size);
    શૂન્ય send_async(ગણના rpc_cmd cmd, std::shared_ptr<અચલ શૂન્ય> આગત, size_t આગત_size);
    શૂન્ય send_async(ગણના rpc_cmd cmd, std::shared_ptr<અચલ શૂન્ય> આગત, size_t આગત_size, શૂન્ય * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    શૂન્ય event_free(ggml_backend_event_t event);
    શૂન્ય event_synchronize(ggml_backend_event_t event);
    શૂન્ય event_record(ggml_backend_event_t event);
    શૂન્ય synchronize();
    શૂન્ય busy_spin_acquire();
    શૂન્ય busy_spin_release();
    શૂન્ય graph_compute(uint32_t device, અચલ ggml_cgraph * cgraph);

    શૂન્ય start(અચલ std::string & endpoint);
    શૂન્ય work();

    ~rpc_dispatcher();

ખાનગી:
    સંરચના rpc_સંદેશ {
        rpc_cmd                       cmd;
        std::shared_ptr<અચલ શૂન્ય>   આગત;
        size_t                        આગત_size;
        શૂન્ય                        * output;
        size_t                        output_size;
        std::promise<શૂન્ય>            completion;
    };
    using rpc_સંદેશ_ptr   = std::shared_ptr<rpc_સંદેશ>;
    using rpc_સંદેશ_queue = message_queue<rpc_સંદેશ_ptr>;
    સંરચના rpc_event {
        rpc_સંદેશ_ptr              સંદેશ;
        std::shared_future<શૂન્ય> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_સંદેશ_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

સ્થિર શૂન્ય rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

શૂન્ય rpc_dispatcher::send(ગણના rpc_cmd cmd, std::shared_ptr<અચલ શૂન્ય> આગત, size_t આગત_size) {
    auto સંદેશ = std::make_shared<rpc_સંદેશ>();
    સંદેશ->cmd = cmd;
    સંદેશ->આગત = આગત;
    સંદેશ->આગત_size = આગત_size;
    સંદેશ->output = nullptr;
    સંદેશ->output_size = 0;
    GGML_ASSERT(queue.ધકેલો(સંદેશ));
    auto future = સંદેશ->completion.get_future();
    future.wait();
}

શૂન્ય rpc_dispatcher::send_async(ગણના rpc_cmd cmd, std::shared_ptr<અચલ શૂન્ય> આગત, size_t આગત_size) {
    auto સંદેશ = std::make_shared<rpc_સંદેશ>();
    સંદેશ->cmd = cmd;
    સંદેશ->આગત = આગત;
    સંદેશ->આગત_size = આગત_size;
    સંદેશ->output = nullptr;
    સંદેશ->output_size = 0;
    GGML_ASSERT(queue.ધકેલો(સંદેશ));
}

શૂન્ય rpc_dispatcher::send(ગણના rpc_cmd cmd, std::shared_ptr<અચલ શૂન્ય> આગત, size_t આગત_size, શૂન્ય * output, size_t output_size) {
    auto સંદેશ = std::make_shared<rpc_સંદેશ>();
    સંદેશ->cmd = cmd;
    સંદેશ->આગત = આગત;
    સંદેશ->આગત_size = આગત_size;
    સંદેશ->output = output;
    સંદેશ->output_size = output_size;
    GGML_ASSERT(queue.ધકેલો(સંદેશ));
    auto future = સંદેશ->completion.get_future();
    future.wait();
}

શૂન્ય rpc_dispatcher::send_async(ગણના rpc_cmd cmd, std::shared_ptr<અચલ શૂન્ય> આગત, size_t આગત_size, શૂન્ય * output, size_t output_size) {
    auto સંદેશ = std::make_shared<rpc_સંદેશ>();
    સંદેશ->cmd = cmd;
    સંદેશ->આગત = આગત;
    સંદેશ->આગત_size = આગત_size;
    સંદેશ->output = output;
    સંદેશ->output_size = output_size;
    GGML_ASSERT(queue.ધકેલો(સંદેશ));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = નવું rpc_event;
    ev->સંદેશ = std::make_shared<rpc_સંદેશ>();
    ev->સંદેશ->cmd = RPC_CMD_NONE;
    ev->sf = ev->સંદેશ->completion.get_future().share();
    GGML_ASSERT(queue.ધકેલો(ev->સંદેશ));
    પાછા નવું ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

શૂન્ય rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

શૂન્ય rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

શૂન્ય rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->સંદેશ = std::make_shared<rpc_સંદેશ>();
    ev->સંદેશ->cmd = RPC_CMD_NONE;
    ev->sf = ev->સંદેશ->completion.get_future().share();
    GGML_ASSERT(queue.ધકેલો(ev->સંદેશ));
}

શૂન્ય rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto સંદેશ = std::make_shared<rpc_સંદેશ>();
    સંદેશ->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ધકેલો(સંદેશ));
    સંદેશ->completion.get_future().wait();
}

શૂન્ય rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

શૂન્ય rpc_dispatcher::busy_spin_release() {
    અચલ unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

શૂન્ય rpc_dispatcher::start(અચલ std::string & endpoint) {
    std::string host;
    પૂર્ણાંક port;
    જો (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    જો (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    જો (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    જો (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = સાચું;
    thread = std::thread(rpc_dispatcher_trampoline, આ);
}

શૂન્ય rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    જ્યારે (running) {
        rpc_સંદેશ_ptr સંદેશ_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        અચલ તાર્કિક spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        જો (spin) {
            જો (!queue.try_કાઢો(&સંદેશ_ptr)) {
                rpc_cpu_relax();
                ચાલુ;
            }
        } નહીંતર જો (!queue.કાઢો(&સંદેશ_ptr)) {
            તોડો;
        }
        જો (સંદેશ_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(સંદેશ_ptr->cmd),
                     rpc_log_format_size(સંદેશ_ptr->આગત_size).c_str(), rpc_log_format_size(સંદેશ_ptr->output_size).c_str());
            અચલ તાર્કિક trace = rpc_log_debug_level() >= 3;
            અચલ auto t0 = std::chrono::steady_clock::now();
            જો (સંદેશ_ptr->output) {
                તાર્કિક status = send_rpc_cmd(sock, સંદેશ_ptr->cmd, સંદેશ_ptr->આગત.get(), સંદેશ_ptr->આગત_size, સંદેશ_ptr->output, સંદેશ_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } નહીંતર {
                તાર્કિક status = send_rpc_cmd(sock, સંદેશ_ptr->cmd, સંદેશ_ptr->આગત.get(), સંદેશ_ptr->આગત_size);
                RPC_STATUS_ASSERT(status);
            }
            જો (trace) {
                અચલ std::chrono::duration<દ્વિગુણ, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(સંદેશ_ptr->cmd), elapsed.count());
            }
        } નહીંતર {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        સંદેશ_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = ખોટું;
    queue.interrupt();
    sock = nullptr;
    જો (thread.joinable()) {
        thread.join();
    }
}

સ્થિર std::shared_ptr<rpc_dispatcher> get_dispatcher(અચલ std::string & endpoint) {
    સ્થિર std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    સ્થિર std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    જો (it != dispatchers.end()) {
        જો (auto dispatcher = it->second.lock()) {
            પાછા dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    પાછા dispatcher;
}

સ્થિર શૂન્ય ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_સંદેશ_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, કદ(*request));
    delete ctx;
}

સ્થિર શૂન્ય * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    જો (ctx->base_ptr != nullptr) {
        પાછા ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_સંદેશ_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_સંદેશ_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, કદ(*request), &response, કદ(response));
    ctx->base_ptr = reinterpret_cast<શૂન્ય *>(response.base_ptr);
    પાછા ctx->base_ptr;
}

સ્થિર તાર્કિક ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    પાછા buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

સ્થિર rpc_tensor serialize_tensor(અચલ ggml_tensor * tensor, અચલ rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    જો (!tensor) {
        memset(&result, 0, કદ(result));
        પાછા result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    જો (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        જો (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } નહીંતર {
            result.buffer = 0;
            result.data = 0;
        }
    } નહીંતર {
        result.buffer = 0;
        result.data   = 0;
    }
    માટે (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    માટે (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / કદ(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    માટે (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, કદ(result.name));
    result.use_ગણતરી = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    પાછા result;
}

સ્થિર ગણના ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    જો (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_સંદેશ_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, કદ(*request));
    }
    પાછા GGML_STATUS_SUCCESS;
}

સ્થિર શૂન્ય ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_સંદેશ_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, કદ(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
સ્થિર std::shared_ptr<uint8_t> serialize_set_tensor(અચલ rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, અચલ શૂન્ય * data, size_t size, size_t & આગત_size) {
    આગત_size = કદ(rpc_tensor) + કદ(cache_flag) + કદ(offset) + size;
    uint8_t * આગત = નવું uint8_t[આગત_size]();
    uint8_t * p = આગત;
    memcpy(p, &rpc_tensor, કદ(rpc_tensor)); p += કદ(rpc_tensor);
    memcpy(p, &cache_flag, કદ(cache_flag)); p += કદ(cache_flag);
    memcpy(p, &offset,     કદ(offset));     p += કદ(offset);
    memcpy(p, data, size);
    પાછા std::shared_ptr<uint8_t>(આગત, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
સ્થિર તાર્કિક rpc_use_hash_cache(અચલ ggml_tensor * tensor, size_t size) {
    પાછા size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

સ્થિર શૂન્ય ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    જો (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_સંદેશ_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((અચલ uint8_t*)data, size);
        rpc_સંદેશ_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, કદ(*request), &response, કદ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        જો (response.result) {
            // the server has the same data, no need to send it
            પાછા;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t આગત_size;
    auto આગત = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, આગત_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, આગત, આગત_size);
}

સ્થિર શૂન્ય ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, અચલ શૂન્ય * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t આગત_size = કદ(rpc_tensor) + 4*કદ(uint64_t) + size*n_copies;
    uint8_t * આગત = નવું uint8_t[આગત_size]();
    uint8_t * dest = આગત;
    memcpy(dest, &rpc_tensor, કદ(rpc_tensor));
    dest += કદ(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, કદ(header));
    dest += કદ(header);
    માટે (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (અચલ અક્ષર *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> આગત_ptr(આગત, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, આગત_ptr, આગત_size);
}

સ્થિર શૂન્ય ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, અચલ ggml_tensor * tensor, શૂન્ય * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_સંદેશ_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    જો (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, કદ(*request), data, size*n_copies);
    } નહીંતર {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, કદ(*request), packed.data(), packed.size());
        માટે (size_t i = 0; i < n_copies; i++) {
            memcpy((અક્ષર *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

સ્થિર શૂન્ય ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, અચલ ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_સંદેશ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, કદ(*request), data, size);
}

સ્થિર તાર્કિક ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, અચલ ggml_tensor * src, ggml_tensor * dst) {
    જો (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        જો (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            પાછા ખોટું;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_સંદેશ_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_સંદેશ_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, કદ(*request), &response, કદ(response));
        પાછા response.result;
    }
    પાછા ખોટું;
}

સ્થિર શૂન્ય ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_સંદેશ_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, કદ(*request));
}

સ્થિર ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
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
    /* .reset           = */ ખાલી,
};

સ્થિર અચલ અક્ષર * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    પાછા buft_ctx->name.c_str();
}

સ્થિર ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_સંદેશ_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_સંદેશ_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, કદ(*request), &response, કદ(response));
    જો (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            નવું ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        પાછા buffer;
    } નહીંતર {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        પાછા nullptr;
    }
}

સ્થિર size_t get_alignment(અચલ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_સંદેશ_get_alignment_req>();
    request->device = device;
    rpc_સંદેશ_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, કદ(*request), &response, કદ(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    પાછા response.alignment;
}

સ્થિર size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    પાછા buft_ctx->alignment;
}

સ્થિર size_t get_max_size(અચલ std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_સંદેશ_get_max_size_req>();
    request->device = device;
    rpc_સંદેશ_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, કદ(*request), &response, કદ(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    પાછા response.max_size;
}

સ્થિર size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    પાછા buft_ctx->max_size;
}

સ્થિર size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    તાર્કિક rpc_get = ખોટું;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    જો (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        અચલ size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        સંરચના alloc_size_cache_ચાવી {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / કદ(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_ચાવી ચાવી = {};
        ચાવી.device = buft_ctx->device;
        ચાવી.type = tensor->type;
        ચાવી.op = tensor->op;
        memcpy(ચાવી.op_params, tensor->op_params, કદ(ચાવી.op_params));
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
            ચાવી.ne[i] = (uint32_t)tensor->ne[i];
            ચાવી.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((અચલ uint8_t *)&ચાવી, કદ(ચાવી));
        cache_hash = fnv_hash((અચલ uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        સ્થિર std::mutex cache_mutex;
        સ્થિર std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            જો (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                પાછા std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_સંદેશ_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_સંદેશ_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, કદ(*request), &response, કદ(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        પાછા std::max<size_t>(response.alloc_size, min_size);
    }

    પાછા ggml_nbytes(tensor);
}

સ્થિર ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ખાલી,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ખાલી,
    /* .is_host             = */ ખાલી,
};

સ્થિર અચલ અક્ષર * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    પાછા rpc_ctx->name.c_str();
}

સ્થિર શૂન્ય ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

સ્થિર શૂન્ય ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    જો (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_સંદેશ_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((અચલ uint8_t*)data, size);
        rpc_સંદેશ_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, કદ(*request), &response, કદ(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        જો (response.result) {
            // the server has the same data, no need to send it
            પાછા;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t આગત_size;
    auto આગત = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, આગત_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, આગત, આગત_size);
}

સ્થિર શૂન્ય ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, અચલ ggml_tensor * tensor, શૂન્ય * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_સંદેશ_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, કદ(*request), data, size);
}

સ્થિર શૂન્ય ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

સ્થિર શૂન્ય add_tensor(ggml_tensor * tensor, અચલ ggml_cgraph * cgraph, અચલ rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    જો (tensor == nullptr) {
        પાછા;
    }
    જો (visited.find(tensor) != visited.end()) {
        પાછા;
    }
    visited.insert(tensor);
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    અચલ size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    જો (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_ગણતરી = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

સ્થિર uint8_t * serialize_graph(uint32_t device, અચલ ggml_cgraph * cgraph, અચલ rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    માટે (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*કદ(uint32_t) + કદ(uint64_t) + n_nodes * કદ(uint64_t) + કદ(uint32_t) + n_tensors * કદ(rpc_tensor);
    uint8_t * output = નવું uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, કદ(device));
    dest += કદ(device);
    memcpy(dest, &cgraph->uid, કદ(cgraph->uid));
    dest += કદ(cgraph->uid);
    memcpy(dest, &n_nodes, કદ(n_nodes));
    dest += કદ(n_nodes);
    માટે (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * કદ(uint64_t), &cgraph->nodes[i], કદ(uint64_t));
    }
    dest += n_nodes * કદ(uint64_t);
    memcpy(dest, &n_tensors, કદ(n_tensors));
    dest += કદ(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * કદ(rpc_tensor));
    પાછા output;
}

શૂન્ય rpc_dispatcher::graph_compute(uint32_t device, અચલ ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    તાર્કિક reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    જો (reuse) {
        auto request = std::make_shared<rpc_સંદેશ_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, કદ(*request));
    } નહીંતર {
        જો (cgraph->uid != 0) {
            જો (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t આગત_size = 0;
        uint8_t * આગત = serialize_graph(device, cgraph, આ, &આગત_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(આગત_size).c_str());
        std::shared_ptr<uint8_t> આગત_ptr(આગત, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, આગત_ptr, આગત_size);
    }
}

સ્થિર ગણના ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    પાછા GGML_STATUS_SUCCESS;
}

સ્થિર શૂન્ય ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

સ્થિર શૂન્ય ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

સ્થિર શૂન્ય ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, અચલ શૂન્ય * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

સ્થિર શૂન્ય ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, અચલ ggml_tensor * tensor, શૂન્ય * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

સ્થિર ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ખાલી,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ ખાલી,
    /* .graph_plan_free         = */ ખાલી,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ખાલી,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ ખાલી,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(અચલ અક્ષર * endpoint, uint32_t device) {
    સ્થિર std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    સ્થિર std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    જો (it != buft_map.end()) {
        પાછા it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = નવું ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = નવું ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    પાછા buft;
}

ggml_backend_t ggml_backend_rpc_init(અચલ અક્ષર * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = નવું ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = નવું ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    પાછા backend;
}

તાર્કિક ggml_backend_is_rpc(ggml_backend_t backend) {
    પાછા backend != ખાલી && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

શૂન્ય ggml_backend_rpc_get_device_memory(અચલ અક્ષર * endpoint, uint32_t device, size_t * મુક્ત, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_સંદેશ_get_device_memory_req>();
    request->device = device;
    rpc_સંદેશ_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, કદ(*request), &response, કદ(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *મુક્ત = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

વર્ગ rpc_server {
જાહેર:
    rpc_server(std::vector<ggml_backend_t> all_backends, અચલ અક્ષર * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    શૂન્ય hello(rpc_સંદેશ_hello_rsp & response);
    તાર્કિક alloc_buffer(અચલ rpc_સંદેશ_alloc_buffer_req & request, rpc_સંદેશ_alloc_buffer_rsp & response);
    તાર્કિક get_alignment(અચલ rpc_સંદેશ_get_alignment_req & request, rpc_સંદેશ_get_alignment_rsp & response);
    તાર્કિક get_max_size(અચલ rpc_સંદેશ_get_max_size_req & request, rpc_સંદેશ_get_max_size_rsp & response);
    તાર્કિક buffer_get_base(અચલ rpc_સંદેશ_buffer_get_base_req & request, rpc_સંદેશ_buffer_get_base_rsp & response);
    તાર્કિક free_buffer(અચલ rpc_સંદેશ_free_buffer_req & request);
    તાર્કિક buffer_clear(અચલ rpc_સંદેશ_buffer_clear_req & request);
    તાર્કિક memset_tensor(અચલ rpc_સંદેશ_memset_tensor_req & request);
    તાર્કિક set_tensor(અચલ std::vector<uint8_t> & આગત);
    તાર્કિક set_tensor_2d(અચલ std::vector<uint8_t> & આગત);
    તાર્કિક set_tensor_hash(અચલ rpc_સંદેશ_set_tensor_hash_req & request, rpc_સંદેશ_set_tensor_hash_rsp & response);
    તાર્કિક get_tensor(અચલ rpc_સંદેશ_get_tensor_req & request, std::vector<uint8_t> & response);
    તાર્કિક get_tensor_2d(અચલ rpc_સંદેશ_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    તાર્કિક copy_tensor(અચલ rpc_સંદેશ_copy_tensor_req & request, rpc_સંદેશ_copy_tensor_rsp & response);
    તાર્કિક graph_compute(અચલ std::vector<uint8_t> & આગત);
    તાર્કિક graph_recompute(અચલ rpc_સંદેશ_graph_recompute_req & request);
    તાર્કિક comm_init(અચલ rpc_સંદેશ_comm_init_req & request, rpc_સંદેશ_comm_init_rsp & response);
    તાર્કિક comm_allreduce(અચલ rpc_સંદેશ_comm_allreduce_req & request);
    તાર્કિક comm_free(અચલ rpc_સંદેશ_comm_free_req & request);
    તાર્કિક init_tensor(અચલ rpc_સંદેશ_init_tensor_req & request);
    તાર્કિક get_alloc_size(અચલ rpc_સંદેશ_get_alloc_size_req & request, rpc_સંદેશ_get_alloc_size_rsp & response);
    તાર્કિક get_device_memory(અચલ rpc_સંદેશ_get_device_memory_req & request, rpc_સંદેશ_get_device_memory_rsp & response);

    સંરચના stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

ખાનગી:
    શૂન્ય sync_all_backends();
    તાર્કિક get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(સંરચના ggml_context * ctx, અચલ rpc_tensor * tensor);
    ggml_tensor * create_ગાંઠ(uint64_t id,
                              સંરચના ggml_context * ctx,
                              અચલ std::unordered_map<uint64_t, અચલ rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, સંરચના ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    સંરચના comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    અચલ અક્ષર * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

શૂન્ય rpc_server::hello(rpc_સંદેશ_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

તાર્કિક rpc_server::get_alloc_size(અચલ rpc_સંદેશ_get_alloc_size_req & request, rpc_સંદેશ_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    જો (dev_id >= backends.size()) {
        પાછા ખોટું;
    }
    ggml_backend_buffer_type_t buft;
    સંરચના ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    જો (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        પાછા ખોટું;
    }
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        જો (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (શૂન્ય*)tensor->buffer, tensor->data);
    જો (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } નહીંતર {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    પાછા સાચું;
}

તાર્કિક rpc_server::alloc_buffer(અચલ rpc_સંદેશ_alloc_buffer_req & request, rpc_સંદેશ_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    જો (dev_id >= backends.size()) {
        પાછા ખોટું;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    જો (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } નહીંતર {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    પાછા સાચું;
}

તાર્કિક rpc_server::get_alignment(અચલ rpc_સંદેશ_get_alignment_req & request, rpc_સંદેશ_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    જો (dev_id >= backends.size()) {
        પાછા ખોટું;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    પાછા સાચું;
}

તાર્કિક rpc_server::get_max_size(અચલ rpc_સંદેશ_get_max_size_req & request, rpc_સંદેશ_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    જો (dev_id >= backends.size()) {
        પાછા ખોટું;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    પાછા સાચું;
}

તાર્કિક rpc_server::buffer_get_base(અચલ rpc_સંદેશ_buffer_get_base_req & request, rpc_સંદેશ_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    જો (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        પાછા ખોટું;
    }
    શૂન્ય * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    પાછા સાચું;
}

તાર્કિક rpc_server::free_buffer(અચલ rpc_સંદેશ_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    જો (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        પાછા ખોટું;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    માટે (auto & sgs : stored_graphs) {
        માટે (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    પાછા સાચું;
}

તાર્કિક rpc_server::buffer_clear(અચલ rpc_સંદેશ_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    જો (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        પાછા ખોટું;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    પાછા સાચું;
}

તાર્કિક rpc_server::memset_tensor(અચલ rpc_સંદેશ_memset_tensor_req & request) {
    sync_all_backends();
    સંરચના ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    જો (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        પાછા ખોટું;
    }

    અચલ uint64_t tensor_size = ggml_nbytes(tensor);
    જો (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        પાછા ખોટું;
    }

    અચલ uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    અચલ uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    જો (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        પાછા ખોટું;
    }
    અચલ uint64_t data_offset = request.tensor.data - buffer_start;
    જો (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        પાછા ખોટું;
    }
    જો (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        પાછા ખોટું;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (શૂન્ય *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    પાછા સાચું;
}

ggml_tensor * rpc_server::deserialize_tensor(સંરચના ggml_context * ctx, અચલ rpc_tensor * tensor) {
    // Validate tensor type before using it
    જો (tensor->type >= GGML_TYPE_ગણતરી) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        પાછા nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    જો (ggml_blck_size((ગણના ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        પાછા nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    જો (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        પાછા nullptr;
    }

    માટે (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    જો (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    જો (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        જો (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            પાછા nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    માટે (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / કદ(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<શૂન્ય *>(tensor->data);
    ggml_set_name(result, tensor->name);
    પાછા result;
}


તાર્કિક rpc_server::set_tensor(અચલ std::vector<uint8_t> & આગત) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    અચલ size_t header_size = કદ(rpc_tensor) + કદ(cache_flag) + કદ(offset);
    જો (આગત.size() < header_size) {
        પાછા ખોટું;
    }
    અચલ rpc_tensor * in_tensor = (અચલ rpc_tensor *)આગત.data();
    memcpy(&cache_flag, આગત.data() + કદ(rpc_tensor), કદ(cache_flag));
    memcpy(&offset,     આગત.data() + કદ(rpc_tensor) + કદ(cache_flag), કદ(offset));
    અચલ size_t size = આગત.size() - header_size;

    સંરચના ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    જો (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        પાછા ખોટું;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (શૂન્ય*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        અચલ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        અચલ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        જો (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            પાછા ખોટું;
        }
    }

    અચલ શૂન્ય * data = આગત.data() + header_size;
    જો (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((અચલ uint8_t*)data, size);
        અક્ષર hash_str[17];
        snprintf(hash_str, કદ(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.લિખ((અચલ અક્ષર *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    પાછા સાચું;
}

તાર્કિક rpc_server::set_tensor_2d(અચલ std::vector<uint8_t> & આગત) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    જો (આગત.size() < કદ(rpc_tensor) + 4*કદ(uint64_t)) {
        પાછા ખોટું;
    }
    અચલ rpc_tensor * in_tensor = (અચલ rpc_tensor *)આગત.data();
    uint64_t header[4];
    memcpy(header, આગત.data() + કદ(rpc_tensor), કદ(header));
    અચલ uint64_t offset   = header[0];
    અચલ uint64_t size     = header[1];
    અચલ uint64_t n_copies = header[2];
    અચલ uint64_t stride   = header[3];

    અચલ uint64_t data_size = આગત.size() - કદ(rpc_tensor) - 4*કદ(uint64_t);
    જો (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        પાછા ખોટું;
    }

    સંરચના ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    જો (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        પાછા ખોટું;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (શૂન્ય*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        જો (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            પાછા ખોટું;
        }
        અચલ uint64_t span = (n_copies - 1)*stride + size;
        અચલ uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        અચલ uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        જો (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            પાછા ખોટું;
        }
        જો (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            પાછા ખોટું;
        }
    }

    અચલ શૂન્ય * data = આગત.data() + કદ(rpc_tensor) + 4*કદ(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    પાછા સાચું;
}

તાર્કિક rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    જો (!cache_dir) {
        પાછા ખોટું;
    }
    અક્ષર hash_str[17];
    snprintf(hash_str, કદ(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    જો (!fs::exists(cache_file, ec)) {
        પાછા ખોટું;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.પઠ((અક્ષર *)data.data(), size);
    પાછા સાચું;
}

તાર્કિક rpc_server::set_tensor_hash(અચલ rpc_સંદેશ_set_tensor_hash_req & request, rpc_સંદેશ_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    સંરચના ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    જો (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        પાછા ખોટું;
    }

    std::vector<uint8_t> cached_file;
    જો (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        પાછા સાચું;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (શૂન્ય*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        અચલ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        અચલ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        જો (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            પાછા ખોટું;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    પાછા સાચું;
}

તાર્કિક rpc_server::init_tensor(અચલ rpc_સંદેશ_init_tensor_req & request) {
    sync_all_backends();
    સંરચના ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    જો (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        પાછા ખોટું;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (શૂન્ય*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    જો (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } નહીંતર {
        જો (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    જો (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        પાછા ખોટું;
    }

    પાછા સાચું;
}

તાર્કિક rpc_server::get_tensor(અચલ rpc_સંદેશ_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    સંરચના ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    જો (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        પાછા ખોટું;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (શૂન્ય*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        અચલ size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        અચલ size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        જો (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                પાછા ખોટું;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    પાછા સાચું;
}

તાર્કિક rpc_server::get_tensor_2d(અચલ rpc_સંદેશ_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    સંરચના ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    જો (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        પાછા ખોટું;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (શૂન્ય*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        જો (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            પાછા ખોટું;
        }
        જો (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            પાછા ખોટું;
        }
        અચલ uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        અચલ uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        અચલ uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        જો (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            પાછા ખોટું;
        }
        જો (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            પાછા ખોટું;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    પાછા સાચું;
}

તાર્કિક rpc_server::copy_tensor(અચલ rpc_સંદેશ_copy_tensor_req & request, rpc_સંદેશ_copy_tensor_rsp & response) {
    sync_all_backends();
    સંરચના ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    જો (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        પાછા ખોટું;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    જો (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        પાછા ખોટું;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    પાછા સાચું;
}

ggml_tensor * rpc_server::create_ગાંઠ(uint64_t id,
                                      સંરચના ggml_context * ctx,
                                      અચલ std::unordered_map<uint64_t, અચલ rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, સંરચના ggml_tensor*> & tensor_map) {
    જો (tensor_map.find(id) != tensor_map.end()) {
        પાછા tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    જો (it_ptr == tensor_ptrs.end()) {
        પાછા nullptr;
    }
    અચલ rpc_tensor * tensor = it_ptr->second;

    સંરચના ggml_tensor * result = deserialize_tensor(ctx, tensor);
    જો (result == nullptr) {
        પાછા nullptr;
    }
    જો (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        પાછા nullptr;
    }
    tensor_map[id] = result;
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        જો (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } નહીંતર {
            result->src[i] = create_ગાંઠ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            જો (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                પાછા nullptr;
            }
        }
    }

    // Handle view_src similarly
    જો (tensor->view_src == 0) {
        result->view_src = nullptr;
    } નહીંતર {
        result->view_src = create_ગાંઠ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        જો (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            પાછા nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    પાછા result;
}

તાર્કિક rpc_server::graph_compute(અચલ std::vector<uint8_t> & આગત) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    જો (આગત.size() < 2*કદ(uint32_t) + કદ(uint64_t)) {
        પાછા ખોટું;
    }
    અચલ uint8_t * src = આગત.data();
    uint32_t device;
    memcpy(&device, src, કદ(device));
    src += કદ(device);
    જો (device >= backends.size()) {
        પાછા ખોટું;
    }
    uint64_t uid;
    memcpy(&uid, src, કદ(uid));
    src += કદ(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, કદ(n_nodes));
    src += કદ(n_nodes);
    જો (આગત.size() < 2*કદ(uint32_t) + કદ(uint64_t) + n_nodes*કદ(uint64_t) + કદ(uint32_t)) {
        પાછા ખોટું;
    }
    અચલ uint64_t * nodes = (અચલ uint64_t *)src;
    src += n_nodes*કદ(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, કદ(n_tensors));
    src += કદ(n_tensors);
    જો (આગત.size() < 2*કદ(uint32_t) + કદ(uint64_t) + n_nodes*કદ(uint64_t) + કદ(uint32_t) + n_tensors*કદ(rpc_tensor)) {
        પાછા ખોટું;
    }
    અચલ rpc_tensor * tensors = (અચલ rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    જો (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, ખોટું);
    જો (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    સંરચના ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    સંરચના ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, ખોટું);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, અચલ rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    માટે (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    અચલ auto t_deserialize = std::chrono::steady_clock::now();
    માટે (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], કદ(id));
        graph->nodes[i] = create_ગાંઠ(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        જો (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            પાછા ખોટું;
        }
        જો (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            અચલ size_t hash_pos = ggml_hash_ઉમેરો(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_ગણતરી;
        }
    }
    અચલ std::chrono::duration<દ્વિગુણ, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    પાછા સાચું;
}

તાર્કિક rpc_server::graph_recompute(અચલ rpc_સંદેશ_graph_recompute_req & request) {
    uint32_t device = request.device;
    જો (device >= backends.size()) {
        પાછા ખોટું;
    }
    auto it = stored_graphs[device].find(request.uid);
    જો (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        પાછા ખોટું;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    પાછા સાચું;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
શૂન્ય rpc_server::sync_all_backends() {
    માટે (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
તાર્કિક rpc_server::comm_init(અચલ rpc_સંદેશ_comm_init_req & request, rpc_સંદેશ_comm_init_rsp & response) {
    response.ok = 0;
    જો (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        પાછા સાચું;
    }
    comm_state & state = comm_states[request.device];
    જો (state.peer != nullptr) {
        response.ok = 1;
        પાછા સાચું;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    જો (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        જો (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            પાછા સાચું;
        }
        state.peer = srv->accept();
        જો (state.peer == nullptr) {
            પાછા સાચું;
        }
        જો (!state.peer->recv_data(remote_caps, કદ(remote_caps))) {
            state.peer = nullptr;
            પાછા સાચું;
        }
        state.peer->get_caps(local_caps);
        જો (!state.peer->send_data(local_caps, કદ(local_caps))) {
            state.peer = nullptr;
            પાછા સાચું;
        }
        state.peer->update_caps(remote_caps);
    } નહીંતર {
        અચલ std::string host(request.host, strnlen(request.host, કદ(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        માટે (પૂર્ણાંક i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            જો (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        જો (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            પાછા સાચું;
        }
        state.peer->get_caps(local_caps);
        જો (!state.peer->send_data(local_caps, કદ(local_caps)) ||
            !state.peer->recv_data(remote_caps, કદ(remote_caps))) {
            state.peer = nullptr;
            પાછા સાચું;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    પાછા સાચું;
}

તાર્કિક rpc_server::comm_allreduce(અચલ rpc_સંદેશ_comm_allreduce_req & request) {
    જો (request.device >= backends.size()) {
        પાછા ખોટું;
    }
    comm_state & state = comm_states[request.device];
    જો (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        પાછા ખોટું;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, ખોટું);
    સંરચના ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ખાલી,
        /*.no_alloc   =*/ સાચું,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    જો (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        પાછા ખોટું;
    }
    અચલ size_t  nbytes = ggml_nbytes(t_dst);
    અચલ int64_t ne     = ggml_nelements(t_dst);
    જો (nbytes == 0) {
        પાછા સાચું;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    અચલ તાર્કિક   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    અચલ size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    અચલ size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    જો (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    અક્ષર * scratch_base = (અક્ષર *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        પાછા t;
    };
    auto new_cpy_ગાંઠ = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        પાછા t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, ખોટું);
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
    જો (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_ગાંઠ(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } નહીંતર {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    જો (state.rank == 0) {
        જો (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            પાછા ખોટું;
        }
    } નહીંતર {
        જો (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            પાછા ખોટું;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    જો (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_ગાંઠ(t_wire_recv, t_peer);
    } નહીંતર {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    જો (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } નહીંતર {
        compute_nodes(t_red, nullptr);
    }
    પાછા સાચું;
}

તાર્કિક rpc_server::comm_free(અચલ rpc_સંદેશ_comm_free_req & request) {
    જો (request.device >= backends.size()) {
        પાછા ખોટું;
    }
    comm_states[request.device] = comm_state();
    પાછા સાચું;
}

તાર્કિક rpc_server::get_device_memory(અચલ rpc_સંદેશ_get_device_memory_req & request, rpc_સંદેશ_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    જો (dev_id >= backends.size()) {
        પાછા ખોટું;
    }
    size_t મુક્ત, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &મુક્ત, &total);
    response.free_mem = મુક્ત;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    પાછા સાચું;
}

rpc_server::~rpc_server() {
    માટે (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

સ્થિર શૂન્ય rpc_serve_client(અચલ std::vector<ggml_backend_t> & backends, અચલ અક્ષર * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    જો (!sock->recv_data(&cmd, 1)) {
        પાછા;
    }
    જો (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        પાછા;
    }

    // Read input_size and validate protocol version
    uint64_t hello_આગત_size;
    જો (!sock->recv_data(&hello_આગત_size, કદ(hello_આગત_size))) {
        પાછા;
    }

    જો (hello_આગત_size != કદ(rpc_સંદેશ_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_આગત_size, કદ(rpc_સંદેશ_hello_req), RPC_PROTO_MAJOR_VERSION);
        પાછા;
    }

    rpc_સંદેશ_hello_req req = {};
    જો (!sock->recv_data(&req, કદ(req))) {
        પાછા;
    }

    rpc_સંદેશ_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    જો (!send_સંદેશ(sock, &rsp, કદ(rsp))) {
        પાછા;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    જ્યારે (સાચું) {
        જો (!sock->recv_data(&cmd, 1)) {
            તોડો;
        }
        જો (cmd >= RPC_CMD_ગણતરી) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            તોડો;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((ગણના rpc_cmd)cmd));
        પસંદગી (cmd) {
            સ્થિતિ RPC_CMD_HELLO: {
                // HELLO command is handled above
                પાછા;
            }
            સ્થિતિ RPC_CMD_DEVICE_ગણતરી: {
                જો (!recv_સંદેશ(sock, nullptr, 0)) {
                    પાછા;
                }
                rpc_સંદેશ_device_ગણતરી_rsp response;
                response.device_ગણતરી = backends.size();
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_ALLOC_BUFFER: {
                rpc_સંદેશ_alloc_buffer_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_alloc_buffer_rsp response;
                જો (!server.alloc_buffer(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_GET_ALLOC_SIZE: {
                rpc_સંદેશ_get_alloc_size_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_get_alloc_size_rsp response;
                જો (!server.get_alloc_size(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_GET_ALIGNMENT: {
                rpc_સંદેશ_get_alignment_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_get_alignment_rsp response;
                જો (!server.get_alignment(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_GET_MAX_SIZE: {
                rpc_સંદેશ_get_max_size_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_get_max_size_rsp response;
                જો (!server.get_max_size(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_BUFFER_GET_BASE: {
                rpc_સંદેશ_buffer_get_base_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_buffer_get_base_rsp response;
                જો (!server.buffer_get_base(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_FREE_BUFFER: {
                rpc_સંદેશ_free_buffer_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                જો (!server.free_buffer(request)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_BUFFER_CLEAR: {
                rpc_સંદેશ_buffer_clear_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                જો (!server.buffer_clear(request)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_MEMSET_TENSOR: {
                rpc_સંદેશ_memset_tensor_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                જો (!server.memset_tensor(request)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> આગત;
                જો (!recv_સંદેશ(sock, આગત)) {
                    પાછા;
                }
                જો (!server.set_tensor(આગત)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> આગત;
                જો (!recv_સંદેશ(sock, આગત)) {
                    પાછા;
                }
                જો (!server.set_tensor_2d(આગત)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_GET_TENSOR_2D: {
                rpc_સંદેશ_get_tensor_2d_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                std::vector<uint8_t> response;
                જો (!server.get_tensor_2d(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, response.data(), response.size())) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_SET_TENSOR_HASH: {
                rpc_સંદેશ_set_tensor_hash_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_set_tensor_hash_rsp response;
                જો (!server.set_tensor_hash(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_INIT_TENSOR: {
                rpc_સંદેશ_init_tensor_req request;
                જો (!recv_સંદેશ(sock, &request,કદ(request))) {
                    પાછા;
                }
                જો (!server.init_tensor(request)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_GET_TENSOR: {
                rpc_સંદેશ_get_tensor_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                std::vector<uint8_t> response;
                જો (!server.get_tensor(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, response.data(), response.size())) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_COPY_TENSOR: {
                rpc_સંદેશ_copy_tensor_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_copy_tensor_rsp response;
                જો (!server.copy_tensor(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> આગત;
                જો (!recv_સંદેશ(sock, આગત)) {
                    પાછા;
                }
                જો (!server.graph_compute(આગત)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_સંદેશ_graph_recompute_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                જો (!server.graph_recompute(request)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_COMM_INIT: {
                rpc_સંદેશ_comm_init_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_comm_init_rsp response;
                જો (!server.comm_init(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_COMM_ALLREDUCE: {
                rpc_સંદેશ_comm_allreduce_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                જો (!server.comm_allreduce(request)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_COMM_FREE: {
                rpc_સંદેશ_comm_free_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                જો (!server.comm_free(request)) {
                    પાછા;
                }
                તોડો;
            }
            સ્થિતિ RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_સંદેશ_get_device_memory_req request;
                જો (!recv_સંદેશ(sock, &request, કદ(request))) {
                    પાછા;
                }
                rpc_સંદેશ_get_device_memory_rsp response;
                જો (!server.get_device_memory(request, response)) {
                    પાછા;
                }
                જો (!send_સંદેશ(sock, &response, કદ(response))) {
                    પાછા;
                }
                તોડો;
            }
            મૂળભૂત: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((ગણના rpc_cmd)cmd));
                પાછા;
            }
        }
    }
}

શૂન્ય ggml_backend_rpc_start_server(અચલ અક્ષર * endpoint, અચલ અક્ષર * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    જો (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        પાછા;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    માટે (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t મુક્ત, total;
        ggml_backend_dev_memory(dev, &મુક્ત, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, મુક્ત / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        જો (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            પાછા;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        જો (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            જો (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    પૂર્ણાંક port;
    જો (!parse_endpoint(endpoint, host, port)) {
        પાછા;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#નહીંતર
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    જો (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        પાછા;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    જો (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        પાછા;
    }
    જ્યારે (સાચું) {
        auto client_socket = server_socket->accept();
        જો (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            પાછા;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    માટે (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

સ્થિર અચલ અક્ષર * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    પાછા ctx->name.c_str();
}

સ્થિર અચલ અક્ષર * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    પાછા ctx->description.c_str();
}

સ્થિર શૂન્ય ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, મુક્ત, total);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    પાછા GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

સ્થિર શૂન્ય ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, સંરચના ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ સાચું,
        /* .host_buffer           = */ ખોટું,
        /* .buffer_from_host_ptr  = */ ખોટું,
        /* .events                = */ સાચું,
        /* .mmap_support          = */ સાચું,
    };
}

સ્થિર ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    પાછા ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    પાછા ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

સ્થિર તાર્કિક ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, અચલ સંરચના ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    પાછા સાચું;
}

સ્થિર તાર્કિક ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    જો (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        પાછા ખોટું;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    પાછા buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

સ્થિર ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    પાછા dispatcher->event_new(dev);
}

સ્થિર શૂન્ય ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

સ્થિર શૂન્ય ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

સ્થિર અચલ સંરચના ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ ખાલી,
    /* .buffer_from_host_ptr = */ ખાલી,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ ખાલી,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

સંરચના ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

સ્થિર અચલ અક્ષર * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    પાછા ctx ? ctx->name.c_str() : "RPC";
}

સ્થિર size_t ggml_backend_rpc_reg_get_device_ગણતરી(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    પાછા ctx ? ctx->devices.size() : 0;
}

સ્થિર ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    જો (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } નહીંતર {
        GGML_ASSERT(index < ctx->devices.size());
        પાછા ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
સંરચના ggml_backend_rpc_comm_context {
    સંરચના rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

સ્થિર શૂન્ય ggml_backend_rpc_comm_free(શૂન્ય * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    જો (comm_ctx == nullptr) {
        પાછા;
    }
    માટે (અચલ auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_સંદેશ_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, કદ(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

સ્થિર શૂન્ય * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    જો (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        પાછા nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    માટે (size_t i = 0; i < n_backends; i++) {
        જો (!ggml_backend_is_rpc(backends[i])) {
            પાછા nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        માટે (અચલ auto & rank : ranks) {
            જો (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                પાછા nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    પૂર્ણાંક port0;
    જો (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        પાછા nullptr;
    }
    અચલ uint32_t comm_port = (uint32_t) port0 + 1000;
    જો (host0.size() >= 64) {
        પાછા nullptr;
    }

    માટે (અચલ auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_સંદેશ_comm_init_rsp> responses(n_backends);
    માટે (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_સંદેશ_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        જો (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, કદ(*request), &responses[i], કદ(responses[i]));
    }
    માટે (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    તાર્કિક ok = સાચું;
    માટે (size_t i = 0; i < n_backends; i++) {
        જો (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = ખોટું;
        }
    }
    જો (!ok) {
        માટે (અચલ auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        પાછા nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    પાછા નવું ggml_backend_rpc_comm_context{std::move(ranks)};
}

સ્થિર તાર્કિક ggml_backend_rpc_comm_allreduce_tensor(શૂન્ય * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    જો (comm_ctx == nullptr) {
        પાછા ખોટું;
    }
    અચલ size_t n_ranks = comm_ctx->ranks.size();
    અચલ int64_t ne = ggml_nelements(tensors[0]);
    જો (ne == 0) {
        પાછા સાચું;
    }
    માટે (size_t i = 0; i < n_ranks; i++) {
        જો (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            પાછા ખોટું;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        જો ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            પાછા ખોટું;
        }
    }
    માટે (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_સંદેશ_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, કદ(*request));
    }
    પાછા સાચું;
}

સ્થિર શૂન્ય * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર * name) {
    જો (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        પાછા (શૂન્ય *)ggml_backend_rpc_add_server;
    }
    જો (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        પાછા (શૂન્ય *)ggml_backend_rpc_start_server;
    }
    જો (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        પાછા (શૂન્ય *)ggml_backend_rpc_comm_init;
    }
    જો (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        પાછા (શૂન્ય *)ggml_backend_rpc_comm_free;
    }
    જો (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        પાછા (શૂન્ય *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    પાછા ખાલી;

    GGML_UNUSED(reg);
}

સ્થિર અચલ સંરચના ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_ગણતરી,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(શૂન્ય) {
    સ્થિર સંરચના ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ ખાલી,
    };

    પાછા &ggml_backend_rpc_reg;
}

સ્થિર uint32_t ggml_backend_rpc_get_device_ગણતરી(અચલ અક્ષર * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_સંદેશ_device_ગણતરી_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_ગણતરી, nullptr, 0, &response, કદ(response));
    પાછા response.device_ગણતરી;
}

સ્થિર અચલ ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_ગણતરી,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(અચલ અક્ષર * endpoint) {
    સ્થિર std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    સ્થિર std::mutex mutex;
    સ્થિર uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    જો (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        પાછા reg_map[endpoint];
    }
    uint32_t dev_ગણતરી = ggml_backend_rpc_get_device_ગણતરી(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_ગણતરી);
    જો (dev_ગણતરી == 0) {
        પાછા nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = નવું ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    માટે (uint32_t ind = 0; ind < dev_ગણતરી; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = નવું ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = નવું ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = નવું ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    પાછા reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

#समावेश "ggml-rpc.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-cpp.h"
#समावेश "transport.h"
#समावेश "log.h"

#समावेश <array>
#समावेश <chrono>
#समावेश <cinttypes>
#समावेश <optional>
#समावेश <string>
#समावेश <thread>
#समावेश <vector>
#समावेश <queue>
#समावेश <condition_variable>
#समावेश <future>
#समावेश <memory>
#समावेश <mutex>
#समावेश <unordered_map>
#समावेश <unordered_set>
#समावेश <cstring>
#समावेश <fstream>
#समावेश <filesystem>
#समावेश <algorithm>
#समावेश <atomic>

namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#घोषणा RPC_STATUS_ASSERT(x) अगर (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#आशय pack(धकेलो, 1)
// ggml_tensor is serialized into rpc_tensor
काष्ठा rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t बफर;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / माप(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t तथ्य;
    अक्षर name[GGML_MAX_NAME];

    int32_t use_गिनती;
};

static_assert(माप(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
क्रमागत rpc_cmd {
    RPC_CMD_ALLOC_बफर = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_आकार,
    RPC_CMD_बफर_GET_BASE,
    RPC_CMD_FREE_बफर,
    RPC_CMD_बफर_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_आकार,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_गिनती,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_गिनती,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

अटल स्थिर अक्षर * rpc_cmd_name(क्रमागत rpc_cmd cmd) {
    चयन (cmd) {
        हाल RPC_CMD_ALLOC_बफर:      वापस "ALLOC_BUFFER";
        हाल RPC_CMD_GET_ALIGNMENT:     वापस "GET_ALIGNMENT";
        हाल RPC_CMD_GET_MAX_आकार:      वापस "GET_MAX_SIZE";
        हाल RPC_CMD_बफर_GET_BASE:   वापस "BUFFER_GET_BASE";
        हाल RPC_CMD_FREE_बफर:       वापस "FREE_BUFFER";
        हाल RPC_CMD_बफर_CLEAR:      वापस "BUFFER_CLEAR";
        हाल RPC_CMD_SET_TENSOR:        वापस "SET_TENSOR";
        हाल RPC_CMD_SET_TENSOR_HASH:   वापस "SET_TENSOR_HASH";
        हाल RPC_CMD_GET_TENSOR:        वापस "GET_TENSOR";
        हाल RPC_CMD_COPY_TENSOR:       वापस "COPY_TENSOR";
        हाल RPC_CMD_GRAPH_COMPUTE:     वापस "GRAPH_COMPUTE";
        हाल RPC_CMD_GET_DEVICE_MEMORY: वापस "GET_DEVICE_MEMORY";
        हाल RPC_CMD_INIT_TENSOR:       वापस "INIT_TENSOR";
        हाल RPC_CMD_GET_ALLOC_आकार:    वापस "GET_ALLOC_SIZE";
        हाल RPC_CMD_HELLO:             वापस "HELLO";
        हाल RPC_CMD_DEVICE_गिनती:      वापस "DEVICE_COUNT";
        हाल RPC_CMD_GRAPH_RECOMPUTE:   वापस "GRAPH_RECOMPUTE";
        हाल RPC_CMD_MEMSET_TENSOR:     वापस "MEMSET_TENSOR";
        हाल RPC_CMD_SET_TENSOR_2D:     वापस "SET_TENSOR_2D";
        हाल RPC_CMD_GET_TENSOR_2D:     वापस "GET_TENSOR_2D";
        हाल RPC_CMD_COMM_INIT:         वापस "COMM_INIT";
        हाल RPC_CMD_COMM_ALLREDUCE:    वापस "COMM_ALLREDUCE";
        हाल RPC_CMD_COMM_FREE:         वापस "COMM_FREE";
        हाल RPC_CMD_NONE:              वापस "NONE";
        शेष:                        वापस "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
स्थिर माप_प्रकार HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
स्थिर माप_प्रकार GRAPH_CACHE_MAX = 1024;

काष्ठा rpc_संदेश_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_आकार];
};

काष्ठा rpc_संदेश_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_आकार];
};

काष्ठा rpc_संदेश_device_गिनती_rsp {
    uint32_t device_गिनती;
};

काष्ठा rpc_संदेश_get_alloc_आकार_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

काष्ठा rpc_संदेश_get_alloc_आकार_rsp {
    uint64_t alloc_आकार;
};

काष्ठा rpc_संदेश_init_tensor_req {
    rpc_tensor tensor;
};

काष्ठा rpc_संदेश_alloc_बफर_req {
    uint32_t device;
    uint64_t size;
};

काष्ठा rpc_संदेश_alloc_बफर_rsp {
    uint64_t remote_ptr;
    uint64_t remote_आकार;
};

काष्ठा rpc_संदेश_get_alignment_req {
    uint32_t device;
};

काष्ठा rpc_संदेश_get_alignment_rsp {
    uint64_t alignment;
};

काष्ठा rpc_संदेश_get_max_आकार_req {
    uint32_t device;
};

काष्ठा rpc_संदेश_get_max_आकार_rsp {
    uint64_t max_size;
};

काष्ठा rpc_संदेश_बफर_get_base_req {
    uint64_t remote_ptr;
};

काष्ठा rpc_संदेश_बफर_get_base_rsp {
    uint64_t base_ptr;
};

काष्ठा rpc_संदेश_free_बफर_req {
    uint64_t remote_ptr;
};

काष्ठा rpc_संदेश_बफर_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

काष्ठा rpc_संदेश_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

काष्ठा rpc_संदेश_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

काष्ठा rpc_संदेश_set_tensor_hash_rsp {
    uint8_t परिणाम;
};

काष्ठा rpc_संदेश_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

काष्ठा rpc_संदेश_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

काष्ठा rpc_संदेश_copy_tensor_rsp {
    uint8_t परिणाम;
};

काष्ठा rpc_संदेश_get_device_memory_req {
    uint32_t device;
};

काष्ठा rpc_संदेश_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t योग_mem;
};

काष्ठा rpc_संदेश_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

काष्ठा rpc_संदेश_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

काष्ठा rpc_संदेश_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    अक्षर     host[64];  // rank > 0: rank 0's host
};

काष्ठा rpc_संदेश_comm_init_rsp {
    uint8_t ok;
};

काष्ठा rpc_संदेश_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

काष्ठा rpc_संदेश_comm_free_req {
    uint32_t device;
};

#आशय pack(निकालो)

// RPC data structures

अटल ggml_guid_t ggml_backend_rpc_guid() {
    अटल ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    वापस &guid;
}

काष्ठा ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

काष्ठा ggml_backend_rpc_बफर_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    माप_प्रकार      alignment;
    माप_प्रकार      max_size;
};

श्रेणी rpc_dispatcher;
काष्ठा ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

काष्ठा ggml_backend_rpc_बफर_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    व्योम                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
अटल uint64_t fnv_hash(स्थिर uint8_t * तथ्य, माप_प्रकार len, uint64_t hash = 0xcbf29ce484222325ULL) {
    स्थिर uint64_t fnv_अभाज्य = 0x100000001b3ULL;

    क्रम (माप_प्रकार i = 0; i < len; ++i) {
        hash ^= तथ्य[i];
        hash *= fnv_अभाज्य;
    }
    वापस hash;
}

अटल bool send_संदेश(socket_ptr sock, स्थिर व्योम * संदेश, माप_प्रकार संदेश_आकार) {
    अगर (!sock->send_तथ्य(&संदेश_आकार, माप(संदेश_आकार))) {
        वापस false;
    }
    अगर (!sock->send_तथ्य(संदेश, संदेश_आकार)) {
        वापस false;
    }
    वापस sock->सफाई();
}

अटल bool recv_संदेश(socket_ptr sock, व्योम * संदेश, माप_प्रकार संदेश_आकार) {
    uint64_t size;
    अगर (!sock->recv_तथ्य(&size, माप(size))) {
        वापस false;
    }
    अगर (size != संदेश_आकार) {
        वापस false;
    }
    वापस sock->recv_तथ्य(संदेश, संदेश_आकार);
}

अटल bool recv_संदेश(socket_ptr sock, std::vector<uint8_t> & निवेश) {
    uint64_t size;
    अगर (!sock->recv_तथ्य(&size, माप(size))) {
        वापस false;
    }
    प्रयास {
        निवेश.resize(size);
    } पकड़ो (स्थिर std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        वापस false;
    }
    वापस sock->recv_तथ्य(निवेश.तथ्य(), size);
}

अटल bool parse_endpoint(स्थिर std::string & endpoint, std::string & host, पूर्णांक & port) {
    माप_प्रकार pos = endpoint.find(':');
    अगर (pos == std::string::npos) {
        वापस false;
    }
    host = endpoint.substr(0, pos);
    प्रयास {
        port = std::stoi(endpoint.substr(pos + 1));
    } पकड़ो (...) {
        वापस false;
    }
    वापस true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
अटल bool send_rpc_cmd(socket_ptr sock, क्रमागत rpc_cmd cmd, स्थिर व्योम * निवेश, माप_प्रकार निवेश_आकार) {
    uint8_t cmd_byte = cmd;
    अगर (!sock->send_तथ्य(&cmd_byte, माप(cmd_byte))) {
        वापस false;
    }
    अगर (!sock->send_तथ्य(&निवेश_आकार, माप(निवेश_आकार))) {
        वापस false;
    }
    अगर (!sock->send_तथ्य(निवेश, निवेश_आकार)) {
        वापस false;
    }
    वापस sock->सफाई();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
अटल bool send_rpc_cmd(socket_ptr sock, क्रमागत rpc_cmd cmd, स्थिर व्योम * निवेश, माप_प्रकार निवेश_आकार, व्योम * output, माप_प्रकार output_आकार) {
    अगर (!send_rpc_cmd(sock, cmd, निवेश, निवेश_आकार)) {
        वापस false;
    }
    uint64_t out_आकार;
    अगर (!sock->recv_तथ्य(&out_आकार, माप(out_आकार))) {
        वापस false;
    }
    अगर (out_आकार != output_आकार) {
        वापस false;
    }
    अगर (!sock->recv_तथ्य(output, output_आकार)) {
        वापस false;
    }
    वापस true;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
अटल constexpr स्वतः RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

अटल अंतरभूत व्योम rpc_cpu_relax() {
#अगर defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ अस्थिर("yield" ::: "memory");
#अन्यथा
    std::this_thread::yield();
#पूर्ण_अगर
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
अटल bool negotiate_hello(स्थिर std::shared_ptr<socket_t> & sock) {
    rpc_संदेश_hello_req request = {};
    rpc_संदेश_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, माप(request), &response, माप(response));
    RPC_STATUS_ASSERT(status);

    अगर (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        वापस false;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    वापस true;
}

ढाँचा <typename T>
श्रेणी message_queue {
खुला:
    message_queue() {}

    bool धकेलो(स्थिर T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        अगर (interrupted) {
            वापस false;
        }
        queue.धकेलो(value);
        cvar.notify_all();
        वापस true;
    }

    bool निकालो(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.रुको(lock, [यह] { वापस !queue.empty() || interrupted; });
        अगर (interrupted) {
            वापस false;
        }
        *out = queue.front();
        queue.निकालो();
        वापस true;
    }

    bool try_निकालो(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        अगर (interrupted || queue.empty()) {
            वापस false;
        }
        *out = queue.front();
        queue.निकालो();
        वापस true;
    }

    व्योम interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = true;
        lock.unlock();
        cvar.notify_all();
    }

निजी:
    bool interrupted = false;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

श्रेणी rpc_dispatcher {
खुला:
    rpc_dispatcher() {
    }

    व्योम send(क्रमागत rpc_cmd cmd, std::shared_ptr<स्थिर व्योम> निवेश, माप_प्रकार निवेश_आकार);
    व्योम send(क्रमागत rpc_cmd cmd, std::shared_ptr<स्थिर व्योम> निवेश, माप_प्रकार निवेश_आकार, व्योम * output, माप_प्रकार output_आकार);
    व्योम send_async(क्रमागत rpc_cmd cmd, std::shared_ptr<स्थिर व्योम> निवेश, माप_प्रकार निवेश_आकार);
    व्योम send_async(क्रमागत rpc_cmd cmd, std::shared_ptr<स्थिर व्योम> निवेश, माप_प्रकार निवेश_आकार, व्योम * output, माप_प्रकार output_आकार);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    व्योम event_free(ggml_backend_event_t event);
    व्योम event_synchronize(ggml_backend_event_t event);
    व्योम event_record(ggml_backend_event_t event);
    व्योम synchronize();
    व्योम busy_spin_acquire();
    व्योम busy_spin_release();
    व्योम graph_compute(uint32_t device, स्थिर ggml_cgraph * cgraph);

    व्योम start(स्थिर std::string & endpoint);
    व्योम work();

    ~rpc_dispatcher();

निजी:
    काष्ठा rpc_संदेश {
        rpc_cmd                       cmd;
        std::shared_ptr<स्थिर व्योम>   निवेश;
        माप_प्रकार                        निवेश_आकार;
        व्योम                        * output;
        माप_प्रकार                        output_आकार;
        std::promise<व्योम>            completion;
    };
    using rpc_संदेश_ptr   = std::shared_ptr<rpc_संदेश>;
    using rpc_संदेश_queue = message_queue<rpc_संदेश_ptr>;
    काष्ठा rpc_event {
        rpc_संदेश_ptr              संदेश;
        std::shared_future<व्योम> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_संदेश_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

अटल व्योम rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

व्योम rpc_dispatcher::send(क्रमागत rpc_cmd cmd, std::shared_ptr<स्थिर व्योम> निवेश, माप_प्रकार निवेश_आकार) {
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = cmd;
    संदेश->निवेश = निवेश;
    संदेश->निवेश_आकार = निवेश_आकार;
    संदेश->output = nullptr;
    संदेश->output_आकार = 0;
    GGML_ASSERT(queue.धकेलो(संदेश));
    स्वतः future = संदेश->completion.get_future();
    future.रुको();
}

व्योम rpc_dispatcher::send_async(क्रमागत rpc_cmd cmd, std::shared_ptr<स्थिर व्योम> निवेश, माप_प्रकार निवेश_आकार) {
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = cmd;
    संदेश->निवेश = निवेश;
    संदेश->निवेश_आकार = निवेश_आकार;
    संदेश->output = nullptr;
    संदेश->output_आकार = 0;
    GGML_ASSERT(queue.धकेलो(संदेश));
}

व्योम rpc_dispatcher::send(क्रमागत rpc_cmd cmd, std::shared_ptr<स्थिर व्योम> निवेश, माप_प्रकार निवेश_आकार, व्योम * output, माप_प्रकार output_आकार) {
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = cmd;
    संदेश->निवेश = निवेश;
    संदेश->निवेश_आकार = निवेश_आकार;
    संदेश->output = output;
    संदेश->output_आकार = output_आकार;
    GGML_ASSERT(queue.धकेलो(संदेश));
    स्वतः future = संदेश->completion.get_future();
    future.रुको();
}

व्योम rpc_dispatcher::send_async(क्रमागत rpc_cmd cmd, std::shared_ptr<स्थिर व्योम> निवेश, माप_प्रकार निवेश_आकार, व्योम * output, माप_प्रकार output_आकार) {
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = cmd;
    संदेश->निवेश = निवेश;
    संदेश->निवेश_आकार = निवेश_आकार;
    संदेश->output = output;
    संदेश->output_आकार = output_आकार;
    GGML_ASSERT(queue.धकेलो(संदेश));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = नया rpc_event;
    ev->संदेश = std::make_shared<rpc_संदेश>();
    ev->संदेश->cmd = RPC_CMD_NONE;
    ev->sf = ev->संदेश->completion.get_future().share();
    GGML_ASSERT(queue.धकेलो(ev->संदेश));
    वापस नया ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

व्योम rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    मिटाओ ev;
}

व्योम rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.रुको();
}

व्योम rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->संदेश = std::make_shared<rpc_संदेश>();
    ev->संदेश->cmd = RPC_CMD_NONE;
    ev->sf = ev->संदेश->completion.get_future().share();
    GGML_ASSERT(queue.धकेलो(ev->संदेश));
}

व्योम rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.धकेलो(संदेश));
    संदेश->completion.get_future().रुको();
}

व्योम rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

व्योम rpc_dispatcher::busy_spin_release() {
    स्थिर अचिन्हित previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

व्योम rpc_dispatcher::start(स्थिर std::string & endpoint) {
    std::string host;
    पूर्णांक port;
    अगर (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    अगर (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    अगर (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    अगर (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = true;
    thread = std::thread(rpc_dispatcher_trampoline, यह);
}

व्योम rpc_dispatcher::work() {
    स्वतः last_cmd = std::chrono::steady_clock::now();
    जबतक (running) {
        rpc_संदेश_ptr संदेश_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        स्थिर bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        अगर (spin) {
            अगर (!queue.try_निकालो(&संदेश_ptr)) {
                rpc_cpu_relax();
                जारी;
            }
        } अन्यथा अगर (!queue.निकालो(&संदेश_ptr)) {
            अवरोध;
        }
        अगर (संदेश_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(संदेश_ptr->cmd),
                     rpc_log_format_आकार(संदेश_ptr->निवेश_आकार).c_str(), rpc_log_format_आकार(संदेश_ptr->output_आकार).c_str());
            स्थिर bool trace = rpc_log_debug_level() >= 3;
            स्थिर स्वतः t0 = std::chrono::steady_clock::now();
            अगर (संदेश_ptr->output) {
                bool status = send_rpc_cmd(sock, संदेश_ptr->cmd, संदेश_ptr->निवेश.लो(), संदेश_ptr->निवेश_आकार, संदेश_ptr->output, संदेश_ptr->output_आकार);
                RPC_STATUS_ASSERT(status);
            } अन्यथा {
                bool status = send_rpc_cmd(sock, संदेश_ptr->cmd, संदेश_ptr->निवेश.लो(), संदेश_ptr->निवेश_आकार);
                RPC_STATUS_ASSERT(status);
            }
            अगर (trace) {
                स्थिर std::chrono::duration<द्विगुन, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(संदेश_ptr->cmd), elapsed.count());
            }
        } अन्यथा {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        संदेश_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = false;
    queue.interrupt();
    sock = nullptr;
    अगर (thread.joinable()) {
        thread.join();
    }
}

अटल std::shared_ptr<rpc_dispatcher> get_dispatcher(स्थिर std::string & endpoint) {
    अटल std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    अटल std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    स्वतः it = dispatchers.find(endpoint);
    अगर (it != dispatchers.end()) {
        अगर (स्वतः dispatcher = it->second.lock()) {
            वापस dispatcher;
        }
    }

    स्वतः dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    वापस dispatcher;
}

अटल व्योम ggml_backend_rpc_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    स्वतः request = std::make_shared<rpc_संदेश_free_बफर_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_बफर, request, माप(*request));
    मिटाओ ctx;
}

अटल व्योम * ggml_backend_rpc_बफर_get_base(ggml_backend_बफर_t बफर) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    अगर (ctx->base_ptr != nullptr) {
        वापस ctx->base_ptr;
    }
    स्वतः request = std::make_shared<rpc_संदेश_बफर_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_संदेश_बफर_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_बफर_GET_BASE, request, माप(*request), &response, माप(response));
    ctx->base_ptr = reinterpret_cast<व्योम *>(response.base_ptr);
    वापस ctx->base_ptr;
}

अटल bool ggml_backend_बफर_is_rpc(ggml_backend_बफर_t बफर) {
    वापस बफर->iface.free_बफर == ggml_backend_rpc_बफर_free_बफर;
}

अटल rpc_tensor serialize_tensor(स्थिर ggml_tensor * tensor, स्थिर rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor परिणाम;
    अगर (!tensor) {
        स_रखो(&परिणाम, 0, माप(परिणाम));
        वापस परिणाम;
    }

    परिणाम.id = reinterpret_cast<uint64_t>(tensor);
    परिणाम.type = tensor->type;
    अगर (tensor->बफर && ggml_backend_बफर_is_rpc(tensor->बफर)) {
        ggml_backend_बफर_t बफर = tensor->बफर;
        ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        अगर (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.लो() == dispatcher)) {
            परिणाम.बफर = ctx->remote_ptr;
            परिणाम.तथ्य = reinterpret_cast<uint64_t>(tensor->तथ्य);
        } अन्यथा {
            परिणाम.बफर = 0;
            परिणाम.तथ्य = 0;
        }
    } अन्यथा {
        परिणाम.बफर = 0;
        परिणाम.तथ्य   = 0;
    }
    क्रम (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम.ne[i] = tensor->ne[i];
        परिणाम.nb[i] = tensor->nb[i];
    }
    परिणाम.op = tensor->op;
    क्रम (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / माप(int32_t); i++) {
        परिणाम.op_params[i] = tensor->op_params[i];
    }
    परिणाम.flags = tensor->flags;
    क्रम (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        परिणाम.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    परिणाम.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    परिणाम.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    स_रखो(परिणाम.name, 0, माप(परिणाम.name));
    परिणाम.use_गिनती = 0;

    snprintf(परिणाम.name, GGML_MAX_NAME, "%s", tensor->name);
    वापस परिणाम;
}

अटल क्रमागत ggml_status ggml_backend_rpc_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    अगर (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        स्वतः request = std::make_shared<rpc_संदेश_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, माप(*request));
    }
    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_rpc_बफर_memset_tensor(
        ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    स्वतः request = std::make_shared<rpc_संदेश_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_आकार(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, माप(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
अटल std::shared_ptr<uint8_t> serialize_set_tensor(स्थिर rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, स्थिर व्योम * तथ्य, माप_प्रकार size, माप_प्रकार & निवेश_आकार) {
    निवेश_आकार = माप(rpc_tensor) + माप(cache_flag) + माप(offset) + size;
    uint8_t * निवेश = नया uint8_t[निवेश_आकार]();
    uint8_t * p = निवेश;
    स_नकल(p, &rpc_tensor, माप(rpc_tensor)); p += माप(rpc_tensor);
    स_नकल(p, &cache_flag, माप(cache_flag)); p += माप(cache_flag);
    स_नकल(p, &offset,     माप(offset));     p += माप(offset);
    स_नकल(p, तथ्य, size);
    वापस std::shared_ptr<uint8_t>(निवेश, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
अटल bool rpc_use_hash_cache(स्थिर ggml_tensor * tensor, माप_प्रकार size) {
    वापस size > HASH_THRESHOLD && tensor->बफर->usage == GGML_BACKEND_बफर_USAGE_WEIGHTS;
}

अटल व्योम ggml_backend_rpc_बफर_set_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_आकार(size).c_str());
    अगर (rpc_use_hash_cache(tensor, size)) {
        स्वतः request = std::make_shared<rpc_संदेश_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((स्थिर uint8_t*)तथ्य, size);
        rpc_संदेश_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, माप(*request), &response, माप(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.परिणाम ? "hit" : "miss");
        अगर (response.परिणाम) {
            // the server has the same data, no need to send it
            वापस;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    माप_प्रकार निवेश_आकार;
    स्वतः निवेश = serialize_set_tensor(rpc_tensor, cache_flag, offset, तथ्य, size, निवेश_आकार);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, निवेश, निवेश_आकार);
}

अटल व्योम ggml_backend_rpc_बफर_set_tensor_2d(ggml_backend_बफर_t बफर, ggml_tensor * tensor, स्थिर व्योम * तथ्य,
        माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_आकार(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    माप_प्रकार निवेश_आकार = माप(rpc_tensor) + 4*माप(uint64_t) + size*n_copies;
    uint8_t * निवेश = नया uint8_t[निवेश_आकार]();
    uint8_t * dest = निवेश;
    स_नकल(dest, &rpc_tensor, माप(rpc_tensor));
    dest += माप(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    स_नकल(dest, header, माप(header));
    dest += माप(header);
    क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
        स_नकल(dest + i*size, (स्थिर अक्षर *)तथ्य + i*stride_तथ्य, size);
    }
    std::shared_ptr<uint8_t> निवेश_ptr(निवेश, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, निवेश_ptr, निवेश_आकार);
}

अटल व्योम ggml_backend_rpc_बफर_get_tensor_2d(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य,
        माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_आकार(size).c_str(), n_copies, stride_tensor);
    स्वतः request = std::make_shared<rpc_संदेश_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    अगर (stride_तथ्य == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, माप(*request), तथ्य, size*n_copies);
    } अन्यथा {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, माप(*request), packed.तथ्य(), packed.size());
        क्रम (माप_प्रकार i = 0; i < n_copies; i++) {
            स_नकल((अक्षर *)तथ्य + i*stride_तथ्य, packed.तथ्य() + i*size, size);
        }
    }
}

अटल व्योम ggml_backend_rpc_बफर_get_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_आकार(size).c_str());
    स्वतः request = std::make_shared<rpc_संदेश_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, माप(*request), तथ्य, size);
}

अटल bool ggml_backend_rpc_बफर_cpy_tensor(ggml_backend_बफर_t बफर, स्थिर ggml_tensor * src, ggml_tensor * dst) {
    अगर (ggml_backend_बफर_is_rpc(src->बफर)) {
        // check if src and dst are on the same server
        ggml_backend_बफर_t src_बफर = src->बफर;
        ggml_backend_rpc_बफर_context * src_ctx = (ggml_backend_rpc_बफर_context *)src_बफर->context;
        ggml_backend_बफर_t dst_बफर = dst->बफर;
        ggml_backend_rpc_बफर_context * dst_ctx = (ggml_backend_rpc_बफर_context *)dst_बफर->context;
        अगर (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            वापस false;
        }
        ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        स्वतः request = std::make_shared<rpc_संदेश_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_संदेश_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, माप(*request), &response, माप(response));
        वापस response.परिणाम;
    }
    वापस false;
}

अटल व्योम ggml_backend_rpc_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    स्वतः request = std::make_shared<rpc_संदेश_बफर_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_बफर_CLEAR, request, माप(*request));
}

अटल ggml_backend_बफर_i ggml_backend_rpc_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_rpc_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_बफर_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_बफर_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_बफर_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_बफर_clear,
    /* .reset           = */ शून्य,
};

अटल स्थिर अक्षर * ggml_backend_rpc_बफर_type_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    वापस buft_ctx->name.c_str();
}

अटल ggml_backend_बफर_t ggml_backend_rpc_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    स्वतः request = std::make_shared<rpc_संदेश_alloc_बफर_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_संदेश_alloc_बफर_rsp response;

    स्वतः dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_बफर, request, माप(*request), &response, माप(response));
    अगर (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_आकार(request->size).c_str(),
                response.remote_ptr, rpc_log_format_आकार(response.remote_आकार).c_str());
        ggml_backend_बफर_t बफर = ggml_backend_बफर_init(buft,
            ggml_backend_rpc_बफर_interface,
            नया ggml_backend_rpc_बफर_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_आकार);
        वापस बफर;
    } अन्यथा {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_आकार(request->size).c_str());
        वापस nullptr;
    }
}

अटल माप_प्रकार get_alignment(स्थिर std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    स्वतः request = std::make_shared<rpc_संदेश_get_alignment_req>();
    request->device = device;
    rpc_संदेश_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, माप(*request), &response, माप(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_आकार(response.alignment).c_str());
    वापस response.alignment;
}

अटल माप_प्रकार ggml_backend_rpc_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    वापस buft_ctx->alignment;
}

अटल माप_प्रकार get_max_आकार(स्थिर std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    स्वतः request = std::make_shared<rpc_संदेश_get_max_आकार_req>();
    request->device = device;
    rpc_संदेश_get_max_आकार_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_आकार, request, माप(*request), &response, माप(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_आकार(response.max_size).c_str());
    वापस response.max_size;
}

अटल माप_प्रकार ggml_backend_rpc_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    वापस buft_ctx->max_size;
}

अटल माप_प्रकार ggml_backend_rpc_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_आकार_may_expand(tensor->op);

    अगर (rpc_get) {
        ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        स्थिर माप_प्रकार min_आकार = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        काष्ठा alloc_आकार_cache_कुंजी {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / माप(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_आकार_cache_कुंजी कुंजी = {};
        कुंजी.device = buft_ctx->device;
        कुंजी.type = tensor->type;
        कुंजी.op = tensor->op;
        स_नकल(कुंजी.op_params, tensor->op_params, माप(कुंजी.op_params));
        क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
            कुंजी.ne[i] = (uint32_t)tensor->ne[i];
            कुंजी.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((स्थिर uint8_t *)&कुंजी, माप(कुंजी));
        cache_hash = fnv_hash((स्थिर uint8_t *)buft_ctx->endpoint.तथ्य(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        अटल std::mutex cache_mutex;
        अटल std::unordered_map<uint64_t, माप_प्रकार> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            स्वतः it = cache.find(cache_hash);
            अगर (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_आकार(it->second).c_str());
                वापस std::max<माप_प्रकार>(it->second, min_आकार);
            }
        }

        स्वतः request = std::make_shared<rpc_संदेश_get_alloc_आकार_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_संदेश_get_alloc_आकार_rsp response;
        स्वतः dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_आकार, request, माप(*request), &response, माप(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_आकार;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_आकार(response.alloc_आकार).c_str());

        वापस std::max<माप_प्रकार>(response.alloc_आकार, min_आकार);
    }

    वापस ggml_nbytes(tensor);
}

अटल ggml_backend_बफर_type_i ggml_backend_rpc_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_बफर_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ शून्य,
    /* .get_alignment       = */ ggml_backend_rpc_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_rpc_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ शून्य,
};

अटल स्थिर अक्षर * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    वापस rpc_ctx->name.c_str();
}

अटल व्योम ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    मिटाओ rpc_ctx;
    मिटाओ backend;
}

अटल व्योम ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_आकार(size).c_str());
    अगर (rpc_use_hash_cache(tensor, size)) {
        स्वतः request = std::make_shared<rpc_संदेश_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((स्थिर uint8_t*)तथ्य, size);
        rpc_संदेश_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, माप(*request), &response, माप(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.परिणाम ? "hit" : "miss");
        अगर (response.परिणाम) {
            // the server has the same data, no need to send it
            वापस;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    माप_प्रकार निवेश_आकार;
    स्वतः निवेश = serialize_set_tensor(rpc_tensor, cache_flag, offset, तथ्य, size, निवेश_आकार);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, निवेश, निवेश_आकार);
}

अटल व्योम ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, स्थिर ggml_tensor * tensor, व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_आकार(size).c_str());
    स्वतः request = std::make_shared<rpc_संदेश_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, माप(*request), तथ्य, size);
}

अटल व्योम ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

अटल व्योम add_tensor(ggml_tensor * tensor, स्थिर ggml_cgraph * cgraph, स्थिर rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    अगर (tensor == nullptr) {
        वापस;
    }
    अगर (visited.find(tensor) != visited.end()) {
        वापस;
    }
    visited.insert(tensor);
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor परिणाम = serialize_tensor(tensor, dispatcher);
    स्थिर माप_प्रकार hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    अगर (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        परिणाम.use_गिनती = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(परिणाम);
}

अटल uint8_t * serialize_graph(uint32_t device, स्थिर ggml_cgraph * cgraph, स्थिर rpc_dispatcher * dispatcher, माप_प्रकार * output_आकार) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    क्रम (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_आकार = 2*माप(uint32_t) + माप(uint64_t) + n_nodes * माप(uint64_t) + माप(uint32_t) + n_tensors * माप(rpc_tensor);
    uint8_t * output = नया uint8_t[*output_आकार]();
    uint8_t * dest = output;
    स_नकल(dest, &device, माप(device));
    dest += माप(device);
    स_नकल(dest, &cgraph->uid, माप(cgraph->uid));
    dest += माप(cgraph->uid);
    स_नकल(dest, &n_nodes, माप(n_nodes));
    dest += माप(n_nodes);
    क्रम (uint32_t i = 0; i < n_nodes; i++) {
        स_नकल(dest + i * माप(uint64_t), &cgraph->nodes[i], माप(uint64_t));
    }
    dest += n_nodes * माप(uint64_t);
    स_नकल(dest, &n_tensors, माप(n_tensors));
    dest += माप(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    स_नकल(out_tensors, tensors.तथ्य(), n_tensors * माप(rpc_tensor));
    वापस output;
}

व्योम rpc_dispatcher::graph_compute(uint32_t device, स्थिर ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    स्वतः & device_graph_uids = graph_uids[device];
    bool reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    अगर (reuse) {
        स्वतः request = std::make_shared<rpc_संदेश_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, माप(*request));
    } अन्यथा {
        अगर (cgraph->uid != 0) {
            अगर (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.सफा();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        माप_प्रकार निवेश_आकार = 0;
        uint8_t * निवेश = serialize_graph(device, cgraph, यह, &निवेश_आकार);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_आकार(निवेश_आकार).c_str());
        std::shared_ptr<uint8_t> निवेश_ptr(निवेश, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, निवेश_ptr, निवेश_आकार);
    }
}

अटल क्रमागत ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

अटल व्योम ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

अटल व्योम ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, स्थिर व्योम * तथ्य,
        माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    ggml_backend_tensor_set_2d(tensor, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य);
    GGML_UNUSED(backend);
}

अटल व्योम ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, स्थिर ggml_tensor * tensor, व्योम * तथ्य,
        माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_तथ्य) {
    ggml_backend_tensor_get_2d(tensor, तथ्य, offset, size, n_copies, stride_tensor, stride_तथ्य);
    GGML_UNUSED(backend);
}

अटल ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ शून्य,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ शून्य,
};

ggml_backend_बफर_type_t ggml_backend_rpc_बफर_type(स्थिर अक्षर * endpoint, uint32_t device) {
    अटल std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    अटल std::unordered_map<std::string, ggml_backend_बफर_type_t> buft_map;
    स्वतः it = buft_map.find(buft_name);
    अगर (it != buft_map.end()) {
        वापस it->second;
    }
    स्वतः dispatcher = get_dispatcher(endpoint);
    माप_प्रकार alignment = get_alignment(dispatcher, device);
    माप_प्रकार max_size = get_max_आकार(dispatcher, device);
    ggml_backend_rpc_बफर_type_context * buft_ctx = नया ggml_backend_rpc_बफर_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    स्वतः reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_बफर_type_t buft = नया ggml_backend_बफर_type {
        /* .iface   = */ ggml_backend_rpc_बफर_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_आकार(alignment).c_str(), rpc_log_format_आकार(max_size).c_str());
    वापस buft;
}

ggml_backend_t ggml_backend_rpc_init(स्थिर अक्षर * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    स्वतः dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = नया ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    स्वतः reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = नया ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    वापस backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

व्योम ggml_backend_rpc_get_device_memory(स्थिर अक्षर * endpoint, uint32_t device, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    स्वतः dispatcher = get_dispatcher(endpoint);
    स्वतः request = std::make_shared<rpc_संदेश_get_device_memory_req>();
    request->device = device;
    rpc_संदेश_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, माप(*request), &response, माप(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_आकार(response.free_mem).c_str(), rpc_log_format_आकार(response.योग_mem).c_str());
    *मुक्त = response.free_mem;
    *योग = response.योग_mem;
}

// RPC server-side implementation

श्रेणी rpc_server {
खुला:
    rpc_server(std::vector<ggml_backend_t> all_backends, स्थिर अक्षर * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    व्योम hello(rpc_संदेश_hello_rsp & response);
    bool alloc_बफर(स्थिर rpc_संदेश_alloc_बफर_req & request, rpc_संदेश_alloc_बफर_rsp & response);
    bool get_alignment(स्थिर rpc_संदेश_get_alignment_req & request, rpc_संदेश_get_alignment_rsp & response);
    bool get_max_आकार(स्थिर rpc_संदेश_get_max_आकार_req & request, rpc_संदेश_get_max_आकार_rsp & response);
    bool बफर_get_base(स्थिर rpc_संदेश_बफर_get_base_req & request, rpc_संदेश_बफर_get_base_rsp & response);
    bool free_बफर(स्थिर rpc_संदेश_free_बफर_req & request);
    bool बफर_clear(स्थिर rpc_संदेश_बफर_clear_req & request);
    bool memset_tensor(स्थिर rpc_संदेश_memset_tensor_req & request);
    bool set_tensor(स्थिर std::vector<uint8_t> & निवेश);
    bool set_tensor_2d(स्थिर std::vector<uint8_t> & निवेश);
    bool set_tensor_hash(स्थिर rpc_संदेश_set_tensor_hash_req & request, rpc_संदेश_set_tensor_hash_rsp & response);
    bool get_tensor(स्थिर rpc_संदेश_get_tensor_req & request, std::vector<uint8_t> & response);
    bool get_tensor_2d(स्थिर rpc_संदेश_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(स्थिर rpc_संदेश_copy_tensor_req & request, rpc_संदेश_copy_tensor_rsp & response);
    bool graph_compute(स्थिर std::vector<uint8_t> & निवेश);
    bool graph_recompute(स्थिर rpc_संदेश_graph_recompute_req & request);
    bool comm_init(स्थिर rpc_संदेश_comm_init_req & request, rpc_संदेश_comm_init_rsp & response);
    bool comm_allreduce(स्थिर rpc_संदेश_comm_allreduce_req & request);
    bool comm_free(स्थिर rpc_संदेश_comm_free_req & request);
    bool init_tensor(स्थिर rpc_संदेश_init_tensor_req & request);
    bool get_alloc_आकार(स्थिर rpc_संदेश_get_alloc_आकार_req & request, rpc_संदेश_get_alloc_आकार_rsp & response);
    bool get_device_memory(स्थिर rpc_संदेश_get_device_memory_req & request, rpc_संदेश_get_device_memory_rsp & response);

    काष्ठा stored_graph {
        std::vector<uint8_t>   बफर;
        ggml_cgraph          * graph;
    };

निजी:
    व्योम sync_all_backends();
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & तथ्य);
    ggml_tensor * deserialize_tensor(काष्ठा ggml_context * ctx, स्थिर rpc_tensor * tensor);
    ggml_tensor * create_गाँठ(uint64_t id,
                              काष्ठा ggml_context * ctx,
                              स्थिर std::unordered_map<uint64_t, स्थिर rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, काष्ठा ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    काष्ठा comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_बफर_ptr scratch;
        माप_प्रकार                  scratch_आकार = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    स्थिर अक्षर * cache_dir;
    std::unordered_set<ggml_backend_बफर_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

व्योम rpc_server::hello(rpc_संदेश_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_आकार(स्थिर rpc_संदेश_get_alloc_आकार_req & request, rpc_संदेश_get_alloc_आकार_rsp & response) {
    uint32_t dev_id = request.device;
    अगर (dev_id >= backends.size()) {
        वापस false;
    }
    ggml_backend_बफर_type_t buft;
    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    अगर (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        वापस false;
    }
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        अगर (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (व्योम*)tensor->बफर, tensor->तथ्य);
    अगर (tensor->बफर == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_बफर_type(backends[dev_id]);
    } अन्यथा {
        buft = tensor->बफर->buft;
    }

    response.alloc_आकार = ggml_backend_buft_get_alloc_आकार(buft, tensor);

    वापस true;
}

bool rpc_server::alloc_बफर(स्थिर rpc_संदेश_alloc_बफर_req & request, rpc_संदेश_alloc_बफर_rsp & response) {
    uint32_t dev_id = request.device;
    अगर (dev_id >= backends.size()) {
        वापस false;
    }
    ggml_backend_बफर_type_t buft = ggml_backend_get_default_बफर_type(backends[dev_id]);
    ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर(buft, request.size);
    response.remote_ptr = 0;
    response.remote_आकार = 0;
    अगर (बफर != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(बफर);
        response.remote_आकार = बफर->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_आकार(request.size).c_str(), response.remote_ptr,
                rpc_log_format_आकार(response.remote_आकार).c_str());
        buffers.insert(बफर);
    } अन्यथा {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_आकार(request.size).c_str());
    }
    वापस true;
}

bool rpc_server::get_alignment(स्थिर rpc_संदेश_get_alignment_req & request, rpc_संदेश_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    अगर (dev_id >= backends.size()) {
        वापस false;
    }
    ggml_backend_बफर_type_t buft = ggml_backend_get_default_बफर_type(backends[dev_id]);
    माप_प्रकार alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_आकार(alignment).c_str());
    response.alignment = alignment;
    वापस true;
}

bool rpc_server::get_max_आकार(स्थिर rpc_संदेश_get_max_आकार_req & request, rpc_संदेश_get_max_आकार_rsp & response) {
    uint32_t dev_id = request.device;
    अगर (dev_id >= backends.size()) {
        वापस false;
    }
    ggml_backend_बफर_type_t buft = ggml_backend_get_default_बफर_type(backends[dev_id]);
    माप_प्रकार max_size = ggml_backend_buft_get_max_आकार(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_आकार(max_size).c_str());
    response.max_size = max_size;
    वापस true;
}

bool rpc_server::बफर_get_base(स्थिर rpc_संदेश_बफर_get_base_req & request, rpc_संदेश_बफर_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_बफर_t बफर = reinterpret_cast<ggml_backend_बफर_t>(request.remote_ptr);
    अगर (buffers.find(बफर) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        वापस false;
    }
    व्योम * base = ggml_backend_बफर_get_base(बफर);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    वापस true;
}

bool rpc_server::free_बफर(स्थिर rpc_संदेश_free_बफर_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_बफर_t बफर = reinterpret_cast<ggml_backend_बफर_t>(request.remote_ptr);
    अगर (buffers.find(बफर) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        वापस false;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    क्रम (स्वतः & sgs : stored_graphs) {
        क्रम (स्वतः & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_बफर_free(बफर);
    buffers.erase(बफर);
    वापस true;
}

bool rpc_server::बफर_clear(स्थिर rpc_संदेश_बफर_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_बफर_t बफर = reinterpret_cast<ggml_backend_बफर_t>(request.remote_ptr);
    अगर (buffers.find(बफर) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        वापस false;
    }
    ggml_backend_बफर_clear(बफर, request.value);
    वापस true;
}

bool rpc_server::memset_tensor(स्थिर rpc_संदेश_memset_tensor_req & request) {
    sync_all_backends();
    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    अगर (tensor == nullptr || tensor->बफर == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        वापस false;
    }

    स्थिर uint64_t tensor_आकार = ggml_nbytes(tensor);
    अगर (request.offset > tensor_आकार || request.size > tensor_आकार - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_आकार);
        वापस false;
    }

    स्थिर uint64_t बफर_start = (uint64_t) ggml_backend_बफर_get_base(tensor->बफर);
    स्थिर uint64_t बफर_आकार = ggml_backend_बफर_get_आकार(tensor->बफर);
    अगर (request.tensor.तथ्य < बफर_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        वापस false;
    }
    स्थिर uint64_t तथ्य_offset = request.tensor.तथ्य - बफर_start;
    अगर (तथ्य_offset > बफर_आकार ||
        request.offset > बफर_आकार - तथ्य_offset ||
        request.size > बफर_आकार - तथ्य_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        वापस false;
    }
    अगर (tensor->बफर->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        वापस false;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (व्योम *) tensor->बफर, tensor->तथ्य, request.offset,
            rpc_log_format_आकार(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    वापस true;
}

ggml_tensor * rpc_server::deserialize_tensor(काष्ठा ggml_context * ctx, स्थिर rpc_tensor * tensor) {
    // Validate tensor type before using it
    अगर (tensor->type >= GGML_TYPE_गिनती) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        वापस nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    अगर (ggml_blck_आकार((क्रमागत ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        वापस nullptr;
    }

    ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    अगर (परिणाम == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        वापस nullptr;
    }

    क्रम (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम->nb[i] = tensor->nb[i];
    }
    परिणाम->बफर = reinterpret_cast<ggml_backend_बफर_t>(tensor->बफर);
    अगर (परिणाम->बफर && buffers.find(परिणाम->बफर) == buffers.end()) {
        परिणाम->बफर = nullptr;
    }

    अगर (परिणाम->बफर && ggml_nelements(परिणाम) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_आकार = (uint64_t) ggml_nbytes(परिणाम);
        uint64_t बफर_start = (uint64_t) ggml_backend_बफर_get_base(परिणाम->बफर);
        uint64_t बफर_आकार = (uint64_t) ggml_backend_बफर_get_आकार(परिणाम->बफर);
        अगर (tensor->तथ्य + tensor_आकार < tensor->तथ्य ||
            tensor->तथ्य < बफर_start || tensor->तथ्य + tensor_आकार > बफर_start + बफर_आकार) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(परिणाम->type),
                      परिणाम->ne[0], परिणाम->ne[1], परिणाम->ne[2], परिणाम->ne[3],
                      tensor->तथ्य, tensor->तथ्य + tensor_आकार, बफर_start, बफर_start + बफर_आकार);
            वापस nullptr;
        }
    }

    परिणाम->op = (ggml_op) tensor->op;
    क्रम (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / माप(int32_t); i++) {
        परिणाम->op_params[i] = tensor->op_params[i];
    }
    परिणाम->flags = tensor->flags;
    परिणाम->तथ्य = reinterpret_cast<व्योम *>(tensor->तथ्य);
    ggml_set_name(परिणाम, tensor->name);
    वापस परिणाम;
}


bool rpc_server::set_tensor(स्थिर std::vector<uint8_t> & निवेश) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    स्थिर माप_प्रकार header_आकार = माप(rpc_tensor) + माप(cache_flag) + माप(offset);
    अगर (निवेश.size() < header_आकार) {
        वापस false;
    }
    स्थिर rpc_tensor * in_tensor = (स्थिर rpc_tensor *)निवेश.तथ्य();
    स_नकल(&cache_flag, निवेश.तथ्य() + माप(rpc_tensor), माप(cache_flag));
    स_नकल(&offset,     निवेश.तथ्य() + माप(rpc_tensor) + माप(cache_flag), माप(offset));
    स्थिर माप_प्रकार size = निवेश.size() - header_आकार;

    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    अगर (tensor == nullptr || tensor->बफर == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        वापस false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (व्योम*)tensor->बफर, tensor->तथ्य, offset,
            rpc_log_format_आकार(size).c_str());

    // sanitize tensor->data
    {
        स्थिर माप_प्रकार p0 = (माप_प्रकार) ggml_backend_बफर_get_base(tensor->बफर);
        स्थिर माप_प्रकार p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        अगर (in_tensor->तथ्य + offset < p0 || in_tensor->तथ्य + offset >= p1 || size > (p1 - in_tensor->तथ्य - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->तथ्य, offset, size, p0, p1);
            वापस false;
        }
    }

    स्थिर व्योम * तथ्य = निवेश.तथ्य() + header_आकार;
    अगर (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((स्थिर uint8_t*)तथ्य, size);
        अक्षर hash_str[17];
        snprintf(hash_str, माप(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::बख_प्रवाह ofs(cache_file, std::पनप::द्विग);
        ofs.ग_लिखो((स्थिर अक्षर *)तथ्य, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, तथ्य, offset, size);
    वापस true;
}

bool rpc_server::set_tensor_2d(स्थिर std::vector<uint8_t> & निवेश) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    अगर (निवेश.size() < माप(rpc_tensor) + 4*माप(uint64_t)) {
        वापस false;
    }
    स्थिर rpc_tensor * in_tensor = (स्थिर rpc_tensor *)निवेश.तथ्य();
    uint64_t header[4];
    स_नकल(header, निवेश.तथ्य() + माप(rpc_tensor), माप(header));
    स्थिर uint64_t offset   = header[0];
    स्थिर uint64_t size     = header[1];
    स्थिर uint64_t n_copies = header[2];
    स्थिर uint64_t stride   = header[3];

    स्थिर uint64_t तथ्य_आकार = निवेश.size() - माप(rpc_tensor) - 4*माप(uint64_t);
    अगर (n_copies == 0 || size == 0 || size > तथ्य_आकार / n_copies || size * n_copies != तथ्य_आकार) {
        वापस false;
    }

    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    अगर (tensor == nullptr || tensor->बफर == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        वापस false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (व्योम*)tensor->बफर, tensor->तथ्य, offset,
            rpc_log_format_आकार(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        अगर (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            वापस false;
        }
        स्थिर uint64_t span = (n_copies - 1)*stride + size;
        स्थिर uint64_t p0 = (uint64_t) ggml_backend_बफर_get_base(tensor->बफर);
        स्थिर uint64_t p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        अगर (in_tensor->तथ्य < p0 || in_tensor->तथ्य > p1 || offset > p1 - in_tensor->तथ्य || span > p1 - in_tensor->तथ्य - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->तथ्य, offset, span, p0, p1);
            वापस false;
        }
        अगर (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            वापस false;
        }
    }

    स्थिर व्योम * तथ्य = निवेश.तथ्य() + माप(rpc_tensor) + 4*माप(uint64_t);
    ggml_backend_tensor_set_2d(tensor, तथ्य, offset, size, n_copies, stride, size);
    वापस true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & तथ्य) {
    अगर (!cache_dir) {
        वापस false;
    }
    अक्षर hash_str[17];
    snprintf(hash_str, माप(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    अगर (!fs::exists(cache_file, ec)) {
        वापस false;
    }
    std::अख_प्रवाह ifs(cache_file, std::पनप::द्विग);
    ifs.ग_स्थान(0, std::पनप::end);
    माप_प्रकार size = ifs.tellg();
    ifs.ग_स्थान(0, std::पनप::beg);
    तथ्य.resize(size);
    ifs.पढ़ो((अक्षर *)तथ्य.तथ्य(), size);
    वापस true;
}

bool rpc_server::set_tensor_hash(स्थिर rpc_संदेश_set_tensor_hash_req & request, rpc_संदेश_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    अगर (tensor == nullptr || tensor->बफर == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        वापस false;
    }

    std::vector<uint8_t> cached_file;
    अगर (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.परिणाम = 0;
        वापस true;
    }
    माप_प्रकार size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (व्योम*)tensor->बफर, tensor->तथ्य, request.offset,
            rpc_log_format_आकार(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        स्थिर माप_प्रकार p0 = (माप_प्रकार) ggml_backend_बफर_get_base(tensor->बफर);
        स्थिर माप_प्रकार p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        अगर (request.tensor.तथ्य + request.offset < p0
         || request.tensor.तथ्य + request.offset >= p1
         || size > (p1 - request.tensor.तथ्य - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.तथ्य, request.offset, size, request.hash, p0, p1);
            वापस false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.तथ्य(), request.offset, size);
    response.परिणाम = 1;
    वापस true;
}

bool rpc_server::init_tensor(स्थिर rpc_संदेश_init_tensor_req & request) {
    sync_all_backends();
    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    अगर (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        वापस false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (व्योम*)tensor->बफर, tensor->तथ्य);
    // Call the backend's buffer_init_tensor function
    ggml_backend_बफर_t बफर = tensor->बफर;
    अगर (बफर && बफर->iface.init_tensor) {
        बफर->iface.init_tensor(बफर, tensor);
    } अन्यथा {
        अगर (!बफर) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    अगर (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        वापस false;
    }

    वापस true;
}

bool rpc_server::get_tensor(स्थिर rpc_संदेश_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    अगर (tensor == nullptr || tensor->बफर == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        वापस false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (व्योम*)tensor->बफर, tensor->तथ्य, request.offset,
             rpc_log_format_आकार(request.size).c_str());

    // sanitize tensor->data
    {
        स्थिर माप_प्रकार p0 = (माप_प्रकार) ggml_backend_बफर_get_base(tensor->बफर);
        स्थिर माप_प्रकार p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        अगर (request.tensor.तथ्य + request.offset < p0 ||
            request.tensor.तथ्य + request.offset >= p1 ||
            request.size > (p1 - request.tensor.तथ्य - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.तथ्य, request.offset, request.size, p0, p1);
                वापस false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.तथ्य(), request.offset, request.size);
    वापस true;
}

bool rpc_server::get_tensor_2d(स्थिर rpc_संदेश_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    अगर (tensor == nullptr || tensor->बफर == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        वापस false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (व्योम*)tensor->बफर, tensor->तथ्य, request.offset,
            rpc_log_format_आकार(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        अगर (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            वापस false;
        }
        अगर (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            वापस false;
        }
        स्थिर uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        स्थिर uint64_t p0 = (uint64_t) ggml_backend_बफर_get_base(tensor->बफर);
        स्थिर uint64_t p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        अगर (request.tensor.तथ्य < p0 || request.tensor.तथ्य > p1 || request.offset > p1 - request.tensor.तथ्य ||
                span > p1 - request.tensor.तथ्य - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.तथ्य, request.offset, span, p0, p1);
            वापस false;
        }
        अगर (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            वापस false;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.तथ्य(), request.offset, request.size, request.n_copies, request.stride, request.size);
    वापस true;
}

bool rpc_server::copy_tensor(स्थिर rpc_संदेश_copy_tensor_req & request, rpc_संदेश_copy_tensor_rsp & response) {
    sync_all_backends();
    काष्ठा ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    अगर (src == nullptr || dst == nullptr || src->बफर == nullptr || dst->बफर == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        वापस false;
    }

    uint64_t src_आकार   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_तथ्य   = (uint64_t) dst->तथ्य;
    uint64_t dst_base   = (uint64_t) ggml_backend_बफर_get_base(dst->बफर);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_बफर_get_आकार(dst->बफर);

    अगर (dst_तथ्य + src_आकार > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_तथ्य,
                         dst_तथ्य + src_आकार,
                         dst_base,
                         dst_base + dst_buf_sz);
        वापस false;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.परिणाम = ggml_backend_बफर_copy_tensor(src, dst);
    वापस true;
}

ggml_tensor * rpc_server::create_गाँठ(uint64_t id,
                                      काष्ठा ggml_context * ctx,
                                      स्थिर std::unordered_map<uint64_t, स्थिर rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, काष्ठा ggml_tensor*> & tensor_map) {
    अगर (tensor_map.find(id) != tensor_map.end()) {
        वापस tensor_map[id];
    }
    // Safely find the tensor pointer
    स्वतः it_ptr = tensor_ptrs.find(id);
    अगर (it_ptr == tensor_ptrs.end()) {
        वापस nullptr;
    }
    स्थिर rpc_tensor * tensor = it_ptr->second;

    काष्ठा ggml_tensor * परिणाम = deserialize_tensor(ctx, tensor);
    अगर (परिणाम == nullptr) {
        वापस nullptr;
    }
    अगर (परिणाम->बफर == nullptr && परिणाम->तथ्य != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        वापस nullptr;
    }
    tensor_map[id] = परिणाम;
    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        अगर (tensor->src[i] == 0) {
            परिणाम->src[i] = nullptr;
        } अन्यथा {
            परिणाम->src[i] = create_गाँठ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            अगर (परिणाम->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                वापस nullptr;
            }
        }
    }

    // Handle view_src similarly
    अगर (tensor->view_src == 0) {
        परिणाम->view_src = nullptr;
    } अन्यथा {
        परिणाम->view_src = create_गाँठ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        अगर (परिणाम->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            वापस nullptr;
        }
    }
    परिणाम->view_offs = tensor->view_offs;
    वापस परिणाम;
}

bool rpc_server::graph_compute(स्थिर std::vector<uint8_t> & निवेश) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    अगर (निवेश.size() < 2*माप(uint32_t) + माप(uint64_t)) {
        वापस false;
    }
    स्थिर uint8_t * src = निवेश.तथ्य();
    uint32_t device;
    स_नकल(&device, src, माप(device));
    src += माप(device);
    अगर (device >= backends.size()) {
        वापस false;
    }
    uint64_t uid;
    स_नकल(&uid, src, माप(uid));
    src += माप(uid);
    uint32_t n_nodes;
    स_नकल(&n_nodes, src, माप(n_nodes));
    src += माप(n_nodes);
    अगर (निवेश.size() < 2*माप(uint32_t) + माप(uint64_t) + n_nodes*माप(uint64_t) + माप(uint32_t)) {
        वापस false;
    }
    स्थिर uint64_t * nodes = (स्थिर uint64_t *)src;
    src += n_nodes*माप(uint64_t);
    uint32_t n_tensors;
    स_नकल(&n_tensors, src, माप(n_tensors));
    src += माप(n_tensors);
    अगर (निवेश.size() < 2*माप(uint32_t) + माप(uint64_t) + n_nodes*माप(uint64_t) + माप(uint32_t) + n_tensors*माप(rpc_tensor)) {
        वापस false;
    }
    स्थिर rpc_tensor * tensors = (स्थिर rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    अगर (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].सफा();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    माप_प्रकार buf_आकार = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    अगर (sg.बफर.size() < buf_आकार) {
        sg.बफर.resize(buf_आकार);
    }
    काष्ठा ggml_init_params params = {
        /*.mem_size   =*/ buf_आकार,
        /*.mem_buffer =*/ sg.बफर.तथ्य(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    काष्ठा ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, स्थिर rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    क्रम (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    स्थिर स्वतः t_deserialize = std::chrono::steady_clock::now();
    क्रम (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        स_नकल(&id, &nodes[i], माप(id));
        graph->nodes[i] = create_गाँठ(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        अगर (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            वापस false;
        }
        अगर (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            स्थिर माप_प्रकार hash_pos = ggml_hash_जोड़ो(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_गिनती;
        }
    }
    स्थिर std::chrono::duration<द्विगुन, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    वापस true;
}

bool rpc_server::graph_recompute(स्थिर rpc_संदेश_graph_recompute_req & request) {
    uint32_t device = request.device;
    अगर (device >= backends.size()) {
        वापस false;
    }
    स्वतः it = stored_graphs[device].find(request.uid);
    अगर (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        वापस false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    वापस true;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
व्योम rpc_server::sync_all_backends() {
    क्रम (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
bool rpc_server::comm_init(स्थिर rpc_संदेश_comm_init_req & request, rpc_संदेश_comm_init_rsp & response) {
    response.ok = 0;
    अगर (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        वापस true;
    }
    comm_state & state = comm_states[request.device];
    अगर (state.peer != nullptr) {
        response.ok = 1;
        वापस true;
    }
    uint8_t local_caps[RPC_CONN_CAPS_आकार] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_आकार] = {};
    अगर (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        अगर (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            वापस true;
        }
        state.peer = srv->accept();
        अगर (state.peer == nullptr) {
            वापस true;
        }
        अगर (!state.peer->recv_तथ्य(remote_caps, माप(remote_caps))) {
            state.peer = nullptr;
            वापस true;
        }
        state.peer->get_caps(local_caps);
        अगर (!state.peer->send_तथ्य(local_caps, माप(local_caps))) {
            state.peer = nullptr;
            वापस true;
        }
        state.peer->update_caps(remote_caps);
    } अन्यथा {
        स्थिर std::string host(request.host, strnlen(request.host, माप(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        क्रम (पूर्णांक i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            अगर (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        अगर (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            वापस true;
        }
        state.peer->get_caps(local_caps);
        अगर (!state.peer->send_तथ्य(local_caps, माप(local_caps)) ||
            !state.peer->recv_तथ्य(remote_caps, माप(remote_caps))) {
            state.peer = nullptr;
            वापस true;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    वापस true;
}

bool rpc_server::comm_allreduce(स्थिर rpc_संदेश_comm_allreduce_req & request) {
    अगर (request.device >= backends.size()) {
        वापस false;
    }
    comm_state & state = comm_states[request.device];
    अगर (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        वापस false;
    }
    ggml_backend_t backend = backends[request.device];

    माप_प्रकार ctx_आकार = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
    काष्ठा ggml_init_params params = {
        /*.mem_size   =*/ ctx_आकार,
        /*.mem_buffer =*/ शून्य,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.लो();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    अगर (t_dst == nullptr || t_dst->बफर == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        वापस false;
    }
    स्थिर माप_प्रकार  nbytes = ggml_nbytes(t_dst);
    स्थिर int64_t ne     = ggml_nelements(t_dst);
    अगर (nbytes == 0) {
        वापस true;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    स्थिर bool   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    स्थिर माप_प्रकार wire_bytes = wire_bf16 ? (माप_प्रकार) ne*2 : nbytes;
    स्थिर माप_प्रकार need       = wire_bf16 ? 2*nbytes : nbytes;
    अगर (state.scratch_आकार < need) {
        state.scratch.reset(ggml_backend_alloc_बफर(backend, need));
        state.scratch_आकार = need;
    }
    अक्षर * scratch_base = (अक्षर *) ggml_backend_बफर_get_base(state.scratch.लो());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    स्वतः new_scratch_tensor = [&](ggml_type type, माप_प्रकार offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->बफर = state.scratch.लो();
        t->तथ्य   = scratch_base + offset;
        वापस t;
    };
    स्वतः new_cpy_गाँठ = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->बफर = dst->बफर;
        t->तथ्य   = dst->तथ्य;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        वापस t;
    };
    स्वतः compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
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
    अगर (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_गाँठ(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.तथ्य(), 0, wire_bytes);
    } अन्यथा {
        ggml_backend_tensor_get(t_dst, state.send_buf.तथ्य(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    अगर (state.rank == 0) {
        अगर (!state.peer->send_तथ्य(state.send_buf.तथ्य(), wire_bytes) || !state.peer->सफाई() ||
            !state.peer->recv_तथ्य(state.recv_buf.तथ्य(), wire_bytes)) {
            वापस false;
        }
    } अन्यथा {
        अगर (!state.peer->recv_तथ्य(state.recv_buf.तथ्य(), wire_bytes) ||
            !state.peer->send_तथ्य(state.send_buf.तथ्य(), wire_bytes) || !state.peer->सफाई()) {
            वापस false;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (माप_प्रकार) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    अगर (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.तथ्य(), 0, wire_bytes);
        t_cast = new_cpy_गाँठ(t_wire_recv, t_peer);
    } अन्यथा {
        ggml_backend_tensor_set(t_peer, state.recv_buf.तथ्य(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->बफर = t_dst->बफर;
    t_red->तथ्य   = t_dst->तथ्य;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    अगर (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } अन्यथा {
        compute_nodes(t_red, nullptr);
    }
    वापस true;
}

bool rpc_server::comm_free(स्थिर rpc_संदेश_comm_free_req & request) {
    अगर (request.device >= backends.size()) {
        वापस false;
    }
    comm_states[request.device] = comm_state();
    वापस true;
}

bool rpc_server::get_device_memory(स्थिर rpc_संदेश_get_device_memory_req & request, rpc_संदेश_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    अगर (dev_id >= backends.size()) {
        वापस false;
    }
    माप_प्रकार मुक्त, योग;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &मुक्त, &योग);
    response.free_mem = मुक्त;
    response.योग_mem = योग;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_आकार(response.free_mem).c_str(), rpc_log_format_आकार(response.योग_mem).c_str());
    वापस true;
}

rpc_server::~rpc_server() {
    क्रम (स्वतः बफर : buffers) {
        ggml_backend_बफर_free(बफर);
    }
}

अटल व्योम rpc_serve_client(स्थिर std::vector<ggml_backend_t> & backends, स्थिर अक्षर * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    अगर (!sock->recv_तथ्य(&cmd, 1)) {
        वापस;
    }
    अगर (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        वापस;
    }

    // Read input_size and validate protocol version
    uint64_t hello_निवेश_आकार;
    अगर (!sock->recv_तथ्य(&hello_निवेश_आकार, माप(hello_निवेश_आकार))) {
        वापस;
    }

    अगर (hello_निवेश_आकार != माप(rpc_संदेश_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (माप_प्रकार)hello_निवेश_आकार, माप(rpc_संदेश_hello_req), RPC_PROTO_MAJOR_VERSION);
        वापस;
    }

    rpc_संदेश_hello_req req = {};
    अगर (!sock->recv_तथ्य(&req, माप(req))) {
        वापस;
    }

    rpc_संदेश_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    अगर (!send_संदेश(sock, &rsp, माप(rsp))) {
        वापस;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    जबतक (true) {
        अगर (!sock->recv_तथ्य(&cmd, 1)) {
            अवरोध;
        }
        अगर (cmd >= RPC_CMD_गिनती) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            अवरोध;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((क्रमागत rpc_cmd)cmd));
        चयन (cmd) {
            हाल RPC_CMD_HELLO: {
                // HELLO command is handled above
                वापस;
            }
            हाल RPC_CMD_DEVICE_गिनती: {
                अगर (!recv_संदेश(sock, nullptr, 0)) {
                    वापस;
                }
                rpc_संदेश_device_गिनती_rsp response;
                response.device_गिनती = backends.size();
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_ALLOC_बफर: {
                rpc_संदेश_alloc_बफर_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_alloc_बफर_rsp response;
                अगर (!server.alloc_बफर(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_GET_ALLOC_आकार: {
                rpc_संदेश_get_alloc_आकार_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_get_alloc_आकार_rsp response;
                अगर (!server.get_alloc_आकार(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_GET_ALIGNMENT: {
                rpc_संदेश_get_alignment_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_get_alignment_rsp response;
                अगर (!server.get_alignment(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_GET_MAX_आकार: {
                rpc_संदेश_get_max_आकार_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_get_max_आकार_rsp response;
                अगर (!server.get_max_आकार(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_बफर_GET_BASE: {
                rpc_संदेश_बफर_get_base_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_बफर_get_base_rsp response;
                अगर (!server.बफर_get_base(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_FREE_बफर: {
                rpc_संदेश_free_बफर_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                अगर (!server.free_बफर(request)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_बफर_CLEAR: {
                rpc_संदेश_बफर_clear_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                अगर (!server.बफर_clear(request)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_MEMSET_TENSOR: {
                rpc_संदेश_memset_tensor_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                अगर (!server.memset_tensor(request)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> निवेश;
                अगर (!recv_संदेश(sock, निवेश)) {
                    वापस;
                }
                अगर (!server.set_tensor(निवेश)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> निवेश;
                अगर (!recv_संदेश(sock, निवेश)) {
                    वापस;
                }
                अगर (!server.set_tensor_2d(निवेश)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_GET_TENSOR_2D: {
                rpc_संदेश_get_tensor_2d_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                std::vector<uint8_t> response;
                अगर (!server.get_tensor_2d(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, response.तथ्य(), response.size())) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_SET_TENSOR_HASH: {
                rpc_संदेश_set_tensor_hash_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_set_tensor_hash_rsp response;
                अगर (!server.set_tensor_hash(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_INIT_TENSOR: {
                rpc_संदेश_init_tensor_req request;
                अगर (!recv_संदेश(sock, &request,माप(request))) {
                    वापस;
                }
                अगर (!server.init_tensor(request)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_GET_TENSOR: {
                rpc_संदेश_get_tensor_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                std::vector<uint8_t> response;
                अगर (!server.get_tensor(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, response.तथ्य(), response.size())) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_COPY_TENSOR: {
                rpc_संदेश_copy_tensor_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_copy_tensor_rsp response;
                अगर (!server.copy_tensor(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> निवेश;
                अगर (!recv_संदेश(sock, निवेश)) {
                    वापस;
                }
                अगर (!server.graph_compute(निवेश)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_संदेश_graph_recompute_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                अगर (!server.graph_recompute(request)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_COMM_INIT: {
                rpc_संदेश_comm_init_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_comm_init_rsp response;
                अगर (!server.comm_init(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_COMM_ALLREDUCE: {
                rpc_संदेश_comm_allreduce_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                अगर (!server.comm_allreduce(request)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_COMM_FREE: {
                rpc_संदेश_comm_free_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                अगर (!server.comm_free(request)) {
                    वापस;
                }
                अवरोध;
            }
            हाल RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_संदेश_get_device_memory_req request;
                अगर (!recv_संदेश(sock, &request, माप(request))) {
                    वापस;
                }
                rpc_संदेश_get_device_memory_rsp response;
                अगर (!server.get_device_memory(request, response)) {
                    वापस;
                }
                अगर (!send_संदेश(sock, &response, माप(response))) {
                    वापस;
                }
                अवरोध;
            }
            शेष: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((क्रमागत rpc_cmd)cmd));
                वापस;
            }
        }
    }
}

व्योम ggml_backend_rpc_start_server(स्थिर अक्षर * endpoint, स्थिर अक्षर * cache_dir,
                                   माप_प्रकार n_threads, माप_प्रकार n_devices, ggml_backend_dev_t * devices) {
    अगर (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        वापस;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    क्रम (माप_प्रकार i = 0; i < n_devices; i++) {
        स्वतः dev = devices[i];
        माप_प्रकार मुक्त, योग;
        ggml_backend_dev_memory(dev, &मुक्त, &योग);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 योग / 1024 / 1024, मुक्त / 1024 / 1024);
        स्वतः backend = ggml_backend_dev_init(dev, nullptr);
        अगर (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            वापस;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        अगर (reg) {
            स्वतः ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            अगर (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    पूर्णांक port;
    अगर (!parse_endpoint(endpoint, host, port)) {
        वापस;
    }

#अगर_घोषित GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#अन्यथा
    LOG_INFO("  transport      : TCP\n");
#पूर्ण_अगर // GGML_RPC_RDMA
    अगर (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        वापस;
    }
    स्वतः server_socket = socket_t::create_server(host.c_str(), port);
    अगर (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        वापस;
    }
    जबतक (true) {
        स्वतः client_socket = server_socket->accept();
        अगर (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            वापस;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    क्रम (स्वतः backend : backends) {
        ggml_backend_free(backend);
    }
}

अटल स्थिर अक्षर * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    वापस ctx->name.c_str();
}

अटल स्थिर अक्षर * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    वापस ctx->description.c_str();
}

अटल व्योम ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, मुक्त, योग);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    वापस GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

अटल व्योम ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, काष्ठा ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_योग);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
    };
}

अटल ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    वापस ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

अटल ggml_backend_बफर_type_t ggml_backend_rpc_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    वापस ggml_backend_rpc_बफर_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

अटल bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, स्थिर काष्ठा ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    वापस true;
}

अटल bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    अगर (!buft || buft->iface.get_name != ggml_backend_rpc_बफर_type_name) {
        वापस false;
    }
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    वापस buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

अटल ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वतः dispatcher = get_dispatcher(ctx->endpoint);
    वापस dispatcher->event_new(dev);
}

अटल व्योम ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वतः dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

अटल व्योम ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वतः dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

अटल स्थिर काष्ठा ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_बफर_type,
    /* .get_host_buffer_type = */ शून्य,
    /* .buffer_from_host_ptr = */ शून्य,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ शून्य,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

काष्ठा ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

अटल स्थिर अक्षर * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    वापस ctx ? ctx->name.c_str() : "RPC";
}

अटल माप_प्रकार ggml_backend_rpc_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    वापस ctx ? ctx->devices.size() : 0;
}

अटल ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    अगर (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } अन्यथा {
        GGML_ASSERT(सूचकांक < ctx->devices.size());
        वापस ctx->devices[सूचकांक];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
काष्ठा ggml_backend_rpc_comm_context {
    काष्ठा rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

अटल व्योम ggml_backend_rpc_comm_free(व्योम * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    अगर (comm_ctx == nullptr) {
        वापस;
    }
    क्रम (स्थिर स्वतः & rank : comm_ctx->ranks) {
        स्वतः request = std::make_shared<rpc_संदेश_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, माप(*request));
        rank.dispatcher->busy_spin_release();
    }
    मिटाओ comm_ctx;
}

अटल व्योम * ggml_backend_rpc_comm_init(ggml_backend_t * backends, माप_प्रकार n_backends) {
    अगर (n_backends != 2 || std::दो_पर्या("GGML_RPC_NO_COMM") != nullptr) {
        वापस nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        अगर (!ggml_backend_is_rpc(backends[i])) {
            वापस nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        क्रम (स्थिर स्वतः & rank : ranks) {
            अगर (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                वापस nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    पूर्णांक port0;
    अगर (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        वापस nullptr;
    }
    स्थिर uint32_t comm_port = (uint32_t) port0 + 1000;
    अगर (host0.size() >= 64) {
        वापस nullptr;
    }

    क्रम (स्थिर स्वतः & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_संदेश_comm_init_rsp> responses(n_backends);
    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः request = std::make_shared<rpc_संदेश_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        अगर (i > 0) {
            स_नकल(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, माप(*request), &responses[i], माप(responses[i]));
    }
    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    bool ok = true;
    क्रम (माप_प्रकार i = 0; i < n_backends; i++) {
        अगर (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = false;
        }
    }
    अगर (!ok) {
        क्रम (स्थिर स्वतः & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        वापस nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    वापस नया ggml_backend_rpc_comm_context{std::move(ranks)};
}

अटल bool ggml_backend_rpc_comm_allreduce_tensor(व्योम * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    अगर (comm_ctx == nullptr) {
        वापस false;
    }
    स्थिर माप_प्रकार n_ranks = comm_ctx->ranks.size();
    स्थिर int64_t ne = ggml_nelements(tensors[0]);
    अगर (ne == 0) {
        वापस true;
    }
    क्रम (माप_प्रकार i = 0; i < n_ranks; i++) {
        अगर (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->बफर == nullptr || !ggml_backend_बफर_is_rpc(tensors[i]->बफर)) {
            वापस false;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        अगर ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            वापस false;
        }
    }
    क्रम (माप_प्रकार i = 0; i < n_ranks; i++) {
        स्वतः request = std::make_shared<rpc_संदेश_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, माप(*request));
    }
    वापस true;
}

अटल व्योम * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर * name) {
    अगर (std::म_भेद(name, "ggml_backend_rpc_add_server") == 0) {
        वापस (व्योम *)ggml_backend_rpc_add_server;
    }
    अगर (std::म_भेद(name, "ggml_backend_rpc_start_server") == 0) {
        वापस (व्योम *)ggml_backend_rpc_start_server;
    }
    अगर (std::म_भेद(name, "ggml_backend_comm_init") == 0) {
        वापस (व्योम *)ggml_backend_rpc_comm_init;
    }
    अगर (std::म_भेद(name, "ggml_backend_comm_free") == 0) {
        वापस (व्योम *)ggml_backend_rpc_comm_free;
    }
    अगर (std::म_भेद(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        वापस (व्योम *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    वापस शून्य;

    GGML_UNUSED(reg);
}

अटल स्थिर काष्ठा ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_गिनती,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(व्योम) {
    अटल काष्ठा ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ शून्य,
    };

    वापस &ggml_backend_rpc_reg;
}

अटल uint32_t ggml_backend_rpc_get_device_गिनती(स्थिर अक्षर * endpoint) {
    स्वतः dispatcher = get_dispatcher(endpoint);
    rpc_संदेश_device_गिनती_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_गिनती, nullptr, 0, &response, माप(response));
    वापस response.device_गिनती;
}

अटल स्थिर ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_गिनती,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(स्थिर अक्षर * endpoint) {
    अटल std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    अटल std::mutex mutex;
    अटल uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    अगर (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        वापस reg_map[endpoint];
    }
    uint32_t dev_गिनती = ggml_backend_rpc_get_device_गिनती(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_गिनती);
    अगर (dev_गिनती == 0) {
        वापस nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = नया ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    क्रम (uint32_t ind = 0; ind < dev_गिनती; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = नया ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = नया ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = नया ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    वापस reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

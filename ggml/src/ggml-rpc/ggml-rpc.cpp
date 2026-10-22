#अन्तर्भाव "ggml-rpc.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-cpp.h"
#अन्तर्भाव "transport.h"
#अन्तर्भाव "log.h"

#अन्तर्भाव <array>
#अन्तर्भाव <chrono>
#अन्तर्भाव <cinttypes>
#अन्तर्भाव <optional>
#अन्तर्भाव <string>
#अन्तर्भाव <thread>
#अन्तर्भाव <vector>
#अन्तर्भाव <queue>
#अन्तर्भाव <condition_variable>
#अन्तर्भाव <future>
#अन्तर्भाव <memory>
#अन्तर्भाव <mutex>
#अन्तर्भाव <unordered_map>
#अन्तर्भाव <unordered_set>
#अन्तर्भाव <cstring>
#अन्तर्भाव <fstream>
#अन्तर्भाव <filesystem>
#अन्तर्भाव <algorithm>
#अन्तर्भाव <atomic>

नामाकाश fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) जर (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(ढकला, 1)
// ggml_tensor is serialized into rpc_tensor
संरचना rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / आकार(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    अक्षर name[GGML_MAX_NAME];

    int32_t use_मोजणी;
};

static_assert(आकार(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
गणना rpc_cmd {
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
    RPC_CMD_DEVICE_मोजणी,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_मोजणी,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

स्थिर स्थिरांक अक्षर * rpc_cmd_name(गणना rpc_cmd cmd) {
    निवड (cmd) {
        स्थिती RPC_CMD_ALLOC_BUFFER:      परत "ALLOC_BUFFER";
        स्थिती RPC_CMD_GET_ALIGNMENT:     परत "GET_ALIGNMENT";
        स्थिती RPC_CMD_GET_MAX_SIZE:      परत "GET_MAX_SIZE";
        स्थिती RPC_CMD_BUFFER_GET_BASE:   परत "BUFFER_GET_BASE";
        स्थिती RPC_CMD_FREE_BUFFER:       परत "FREE_BUFFER";
        स्थिती RPC_CMD_BUFFER_CLEAR:      परत "BUFFER_CLEAR";
        स्थिती RPC_CMD_SET_TENSOR:        परत "SET_TENSOR";
        स्थिती RPC_CMD_SET_TENSOR_HASH:   परत "SET_TENSOR_HASH";
        स्थिती RPC_CMD_GET_TENSOR:        परत "GET_TENSOR";
        स्थिती RPC_CMD_COPY_TENSOR:       परत "COPY_TENSOR";
        स्थिती RPC_CMD_GRAPH_COMPUTE:     परत "GRAPH_COMPUTE";
        स्थिती RPC_CMD_GET_DEVICE_MEMORY: परत "GET_DEVICE_MEMORY";
        स्थिती RPC_CMD_INIT_TENSOR:       परत "INIT_TENSOR";
        स्थिती RPC_CMD_GET_ALLOC_SIZE:    परत "GET_ALLOC_SIZE";
        स्थिती RPC_CMD_HELLO:             परत "HELLO";
        स्थिती RPC_CMD_DEVICE_मोजणी:      परत "DEVICE_COUNT";
        स्थिती RPC_CMD_GRAPH_RECOMPUTE:   परत "GRAPH_RECOMPUTE";
        स्थिती RPC_CMD_MEMSET_TENSOR:     परत "MEMSET_TENSOR";
        स्थिती RPC_CMD_SET_TENSOR_2D:     परत "SET_TENSOR_2D";
        स्थिती RPC_CMD_GET_TENSOR_2D:     परत "GET_TENSOR_2D";
        स्थिती RPC_CMD_COMM_INIT:         परत "COMM_INIT";
        स्थिती RPC_CMD_COMM_ALLREDUCE:    परत "COMM_ALLREDUCE";
        स्थिती RPC_CMD_COMM_FREE:         परत "COMM_FREE";
        स्थिती RPC_CMD_NONE:              परत "NONE";
        पूर्वनिर्धारित:                        परत "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
स्थिरांक माप_प्रकार HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
स्थिरांक माप_प्रकार GRAPH_CACHE_MAX = 1024;

संरचना rpc_संदेश_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

संरचना rpc_संदेश_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

संरचना rpc_संदेश_device_मोजणी_rsp {
    uint32_t device_मोजणी;
};

संरचना rpc_संदेश_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

संरचना rpc_संदेश_get_alloc_size_rsp {
    uint64_t alloc_size;
};

संरचना rpc_संदेश_init_tensor_req {
    rpc_tensor tensor;
};

संरचना rpc_संदेश_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

संरचना rpc_संदेश_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

संरचना rpc_संदेश_get_alignment_req {
    uint32_t device;
};

संरचना rpc_संदेश_get_alignment_rsp {
    uint64_t alignment;
};

संरचना rpc_संदेश_get_max_size_req {
    uint32_t device;
};

संरचना rpc_संदेश_get_max_size_rsp {
    uint64_t max_size;
};

संरचना rpc_संदेश_buffer_get_base_req {
    uint64_t remote_ptr;
};

संरचना rpc_संदेश_buffer_get_base_rsp {
    uint64_t base_ptr;
};

संरचना rpc_संदेश_free_buffer_req {
    uint64_t remote_ptr;
};

संरचना rpc_संदेश_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

संरचना rpc_संदेश_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

संरचना rpc_संदेश_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

संरचना rpc_संदेश_set_tensor_hash_rsp {
    uint8_t result;
};

संरचना rpc_संदेश_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

संरचना rpc_संदेश_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

संरचना rpc_संदेश_copy_tensor_rsp {
    uint8_t result;
};

संरचना rpc_संदेश_get_device_memory_req {
    uint32_t device;
};

संरचना rpc_संदेश_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

संरचना rpc_संदेश_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

संरचना rpc_संदेश_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

संरचना rpc_संदेश_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    अक्षर     host[64];  // rank > 0: rank 0's host
};

संरचना rpc_संदेश_comm_init_rsp {
    uint8_t ok;
};

संरचना rpc_संदेश_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

संरचना rpc_संदेश_comm_free_req {
    uint32_t device;
};

#pragma pack(काढा)

// RPC data structures

स्थिर ggml_guid_t ggml_backend_rpc_guid() {
    स्थिर ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    परत &guid;
}

संरचना ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

संरचना ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    माप_प्रकार      alignment;
    माप_प्रकार      max_size;
};

वर्ग rpc_dispatcher;
संरचना ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

संरचना ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    शून्य                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
स्थिर uint64_t fnv_hash(स्थिरांक uint8_t * data, माप_प्रकार len, uint64_t hash = 0xcbf29ce484222325ULL) {
    स्थिरांक uint64_t fnv_मूळसंख्या = 0x100000001b3ULL;

    साठी (माप_प्रकार i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_मूळसंख्या;
    }
    परत hash;
}

स्थिर तार्किक send_संदेश(socket_ptr sock, स्थिरांक शून्य * संदेश, माप_प्रकार संदेश_size) {
    जर (!sock->send_data(&संदेश_size, आकार(संदेश_size))) {
        परत असत्य;
    }
    जर (!sock->send_data(संदेश, संदेश_size)) {
        परत असत्य;
    }
    परत sock->flush();
}

स्थिर तार्किक recv_संदेश(socket_ptr sock, शून्य * संदेश, माप_प्रकार संदेश_size) {
    uint64_t size;
    जर (!sock->recv_data(&size, आकार(size))) {
        परत असत्य;
    }
    जर (size != संदेश_size) {
        परत असत्य;
    }
    परत sock->recv_data(संदेश, संदेश_size);
}

स्थिर तार्किक recv_संदेश(socket_ptr sock, std::vector<uint8_t> & आदान) {
    uint64_t size;
    जर (!sock->recv_data(&size, आकार(size))) {
        परत असत्य;
    }
    प्रयत्न {
        आदान.resize(size);
    } पकडा (स्थिरांक std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        परत असत्य;
    }
    परत sock->recv_data(आदान.data(), size);
}

स्थिर तार्किक parse_endpoint(स्थिरांक std::string & endpoint, std::string & host, पूर्णांक & port) {
    माप_प्रकार pos = endpoint.find(':');
    जर (pos == std::string::npos) {
        परत असत्य;
    }
    host = endpoint.substr(0, pos);
    प्रयत्न {
        port = std::stoi(endpoint.substr(pos + 1));
    } पकडा (...) {
        परत असत्य;
    }
    परत सत्य;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
स्थिर तार्किक send_rpc_cmd(socket_ptr sock, गणना rpc_cmd cmd, स्थिरांक शून्य * आदान, माप_प्रकार आदान_size) {
    uint8_t cmd_byte = cmd;
    जर (!sock->send_data(&cmd_byte, आकार(cmd_byte))) {
        परत असत्य;
    }
    जर (!sock->send_data(&आदान_size, आकार(आदान_size))) {
        परत असत्य;
    }
    जर (!sock->send_data(आदान, आदान_size)) {
        परत असत्य;
    }
    परत sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
स्थिर तार्किक send_rpc_cmd(socket_ptr sock, गणना rpc_cmd cmd, स्थिरांक शून्य * आदान, माप_प्रकार आदान_size, शून्य * output, माप_प्रकार output_size) {
    जर (!send_rpc_cmd(sock, cmd, आदान, आदान_size)) {
        परत असत्य;
    }
    uint64_t out_size;
    जर (!sock->recv_data(&out_size, आकार(out_size))) {
        परत असत्य;
    }
    जर (out_size != output_size) {
        परत असत्य;
    }
    जर (!sock->recv_data(output, output_size)) {
        परत असत्य;
    }
    परत सत्य;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
स्थिर constexpr स्वतः RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

स्थिर ओळीत शून्य rpc_cpu_relax() {
#जर defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ अस्थिर("yield" ::: "memory");
#अन्यथा
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
स्थिर तार्किक negotiate_hello(स्थिरांक std::shared_ptr<socket_t> & sock) {
    rpc_संदेश_hello_req request = {};
    rpc_संदेश_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    तार्किक status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, आकार(request), &response, आकार(response));
    RPC_STATUS_ASSERT(status);

    जर (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        परत असत्य;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    परत सत्य;
}

ढाँचा <typename T>
वर्ग message_queue {
सार्वजनिक:
    message_queue() {}

    तार्किक ढकला(स्थिरांक T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        जर (interrupted) {
            परत असत्य;
        }
        queue.ढकला(value);
        cvar.notify_all();
        परत सत्य;
    }

    तार्किक काढा(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.रुको(lock, [हे] { परत !queue.empty() || interrupted; });
        जर (interrupted) {
            परत असत्य;
        }
        *out = queue.front();
        queue.काढा();
        परत सत्य;
    }

    तार्किक try_काढा(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        जर (interrupted || queue.empty()) {
            परत असत्य;
        }
        *out = queue.front();
        queue.काढा();
        परत सत्य;
    }

    शून्य interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = सत्य;
        lock.unlock();
        cvar.notify_all();
    }

खाजगी:
    तार्किक interrupted = असत्य;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

वर्ग rpc_dispatcher {
सार्वजनिक:
    rpc_dispatcher() {
    }

    शून्य send(गणना rpc_cmd cmd, std::shared_ptr<स्थिरांक शून्य> आदान, माप_प्रकार आदान_size);
    शून्य send(गणना rpc_cmd cmd, std::shared_ptr<स्थिरांक शून्य> आदान, माप_प्रकार आदान_size, शून्य * output, माप_प्रकार output_size);
    शून्य send_async(गणना rpc_cmd cmd, std::shared_ptr<स्थिरांक शून्य> आदान, माप_प्रकार आदान_size);
    शून्य send_async(गणना rpc_cmd cmd, std::shared_ptr<स्थिरांक शून्य> आदान, माप_प्रकार आदान_size, शून्य * output, माप_प्रकार output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    शून्य event_free(ggml_backend_event_t event);
    शून्य event_synchronize(ggml_backend_event_t event);
    शून्य event_record(ggml_backend_event_t event);
    शून्य synchronize();
    शून्य busy_spin_acquire();
    शून्य busy_spin_release();
    शून्य graph_compute(uint32_t device, स्थिरांक ggml_cgraph * cgraph);

    शून्य start(स्थिरांक std::string & endpoint);
    शून्य work();

    ~rpc_dispatcher();

खाजगी:
    संरचना rpc_संदेश {
        rpc_cmd                       cmd;
        std::shared_ptr<स्थिरांक शून्य>   आदान;
        माप_प्रकार                        आदान_size;
        शून्य                        * output;
        माप_प्रकार                        output_size;
        std::promise<शून्य>            completion;
    };
    using rpc_संदेश_ptr   = std::shared_ptr<rpc_संदेश>;
    using rpc_संदेश_queue = message_queue<rpc_संदेश_ptr>;
    संरचना rpc_event {
        rpc_संदेश_ptr              संदेश;
        std::shared_future<शून्य> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_संदेश_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

स्थिर शून्य rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

शून्य rpc_dispatcher::send(गणना rpc_cmd cmd, std::shared_ptr<स्थिरांक शून्य> आदान, माप_प्रकार आदान_size) {
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = cmd;
    संदेश->आदान = आदान;
    संदेश->आदान_size = आदान_size;
    संदेश->output = nullptr;
    संदेश->output_size = 0;
    GGML_ASSERT(queue.ढकला(संदेश));
    स्वतः future = संदेश->completion.get_future();
    future.रुको();
}

शून्य rpc_dispatcher::send_async(गणना rpc_cmd cmd, std::shared_ptr<स्थिरांक शून्य> आदान, माप_प्रकार आदान_size) {
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = cmd;
    संदेश->आदान = आदान;
    संदेश->आदान_size = आदान_size;
    संदेश->output = nullptr;
    संदेश->output_size = 0;
    GGML_ASSERT(queue.ढकला(संदेश));
}

शून्य rpc_dispatcher::send(गणना rpc_cmd cmd, std::shared_ptr<स्थिरांक शून्य> आदान, माप_प्रकार आदान_size, शून्य * output, माप_प्रकार output_size) {
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = cmd;
    संदेश->आदान = आदान;
    संदेश->आदान_size = आदान_size;
    संदेश->output = output;
    संदेश->output_size = output_size;
    GGML_ASSERT(queue.ढकला(संदेश));
    स्वतः future = संदेश->completion.get_future();
    future.रुको();
}

शून्य rpc_dispatcher::send_async(गणना rpc_cmd cmd, std::shared_ptr<स्थिरांक शून्य> आदान, माप_प्रकार आदान_size, शून्य * output, माप_प्रकार output_size) {
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = cmd;
    संदेश->आदान = आदान;
    संदेश->आदान_size = आदान_size;
    संदेश->output = output;
    संदेश->output_size = output_size;
    GGML_ASSERT(queue.ढकला(संदेश));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = नवीन rpc_event;
    ev->संदेश = std::make_shared<rpc_संदेश>();
    ev->संदेश->cmd = RPC_CMD_NONE;
    ev->sf = ev->संदेश->completion.get_future().share();
    GGML_ASSERT(queue.ढकला(ev->संदेश));
    परत नवीन ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

शून्य rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

शून्य rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.रुको();
}

शून्य rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->संदेश = std::make_shared<rpc_संदेश>();
    ev->संदेश->cmd = RPC_CMD_NONE;
    ev->sf = ev->संदेश->completion.get_future().share();
    GGML_ASSERT(queue.ढकला(ev->संदेश));
}

शून्य rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    स्वतः संदेश = std::make_shared<rpc_संदेश>();
    संदेश->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ढकला(संदेश));
    संदेश->completion.get_future().रुको();
}

शून्य rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

शून्य rpc_dispatcher::busy_spin_release() {
    स्थिरांक अचिन्हित previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

शून्य rpc_dispatcher::start(स्थिरांक std::string & endpoint) {
    std::string host;
    पूर्णांक port;
    जर (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    जर (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    जर (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    जर (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = सत्य;
    thread = std::thread(rpc_dispatcher_trampoline, हे);
}

शून्य rpc_dispatcher::work() {
    स्वतः last_cmd = std::chrono::steady_clock::now();
    पर्यंत (running) {
        rpc_संदेश_ptr संदेश_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        स्थिरांक तार्किक spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        जर (spin) {
            जर (!queue.try_काढा(&संदेश_ptr)) {
                rpc_cpu_relax();
                सुरूठेवा;
            }
        } अन्यथा जर (!queue.काढा(&संदेश_ptr)) {
            थांबा;
        }
        जर (संदेश_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(संदेश_ptr->cmd),
                     rpc_log_format_size(संदेश_ptr->आदान_size).c_str(), rpc_log_format_size(संदेश_ptr->output_size).c_str());
            स्थिरांक तार्किक trace = rpc_log_debug_level() >= 3;
            स्थिरांक स्वतः t0 = std::chrono::steady_clock::now();
            जर (संदेश_ptr->output) {
                तार्किक status = send_rpc_cmd(sock, संदेश_ptr->cmd, संदेश_ptr->आदान.get(), संदेश_ptr->आदान_size, संदेश_ptr->output, संदेश_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } अन्यथा {
                तार्किक status = send_rpc_cmd(sock, संदेश_ptr->cmd, संदेश_ptr->आदान.get(), संदेश_ptr->आदान_size);
                RPC_STATUS_ASSERT(status);
            }
            जर (trace) {
                स्थिरांक std::chrono::duration<द्विगुण, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
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
    running = असत्य;
    queue.interrupt();
    sock = nullptr;
    जर (thread.joinable()) {
        thread.join();
    }
}

स्थिर std::shared_ptr<rpc_dispatcher> get_dispatcher(स्थिरांक std::string & endpoint) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    स्थिर std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    स्वतः it = dispatchers.find(endpoint);
    जर (it != dispatchers.end()) {
        जर (स्वतः dispatcher = it->second.lock()) {
            परत dispatcher;
        }
    }

    स्वतः dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    परत dispatcher;
}

स्थिर शून्य ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    स्वतः request = std::make_shared<rpc_संदेश_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, आकार(*request));
    delete ctx;
}

स्थिर शून्य * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    जर (ctx->base_ptr != nullptr) {
        परत ctx->base_ptr;
    }
    स्वतः request = std::make_shared<rpc_संदेश_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_संदेश_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, आकार(*request), &response, आकार(response));
    ctx->base_ptr = reinterpret_cast<शून्य *>(response.base_ptr);
    परत ctx->base_ptr;
}

स्थिर तार्किक ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    परत buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

स्थिर rpc_tensor serialize_tensor(स्थिरांक ggml_tensor * tensor, स्थिरांक rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    जर (!tensor) {
        स_रखो(&result, 0, आकार(result));
        परत result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    जर (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        जर (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } अन्यथा {
            result.buffer = 0;
            result.data = 0;
        }
    } अन्यथा {
        result.buffer = 0;
        result.data   = 0;
    }
    साठी (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    साठी (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / आकार(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    साठी (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    स_रखो(result.name, 0, आकार(result.name));
    result.use_मोजणी = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    परत result;
}

स्थिर गणना ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    जर (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        स्वतः request = std::make_shared<rpc_संदेश_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, आकार(*request));
    }
    परत GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    स्वतः request = std::make_shared<rpc_संदेश_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, आकार(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
स्थिर std::shared_ptr<uint8_t> serialize_set_tensor(स्थिरांक rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, स्थिरांक शून्य * data, माप_प्रकार size, माप_प्रकार & आदान_size) {
    आदान_size = आकार(rpc_tensor) + आकार(cache_flag) + आकार(offset) + size;
    uint8_t * आदान = नवीन uint8_t[आदान_size]();
    uint8_t * p = आदान;
    स_नकल(p, &rpc_tensor, आकार(rpc_tensor)); p += आकार(rpc_tensor);
    स_नकल(p, &cache_flag, आकार(cache_flag)); p += आकार(cache_flag);
    स_नकल(p, &offset,     आकार(offset));     p += आकार(offset);
    स_नकल(p, data, size);
    परत std::shared_ptr<uint8_t>(आदान, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
स्थिर तार्किक rpc_use_hash_cache(स्थिरांक ggml_tensor * tensor, माप_प्रकार size) {
    परत size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

स्थिर शून्य ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    जर (rpc_use_hash_cache(tensor, size)) {
        स्वतः request = std::make_shared<rpc_संदेश_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((स्थिरांक uint8_t*)data, size);
        rpc_संदेश_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, आकार(*request), &response, आकार(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        जर (response.result) {
            // the server has the same data, no need to send it
            परत;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    माप_प्रकार आदान_size;
    स्वतः आदान = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, आदान_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, आदान, आदान_size);
}

स्थिर शून्य ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, स्थिरांक शून्य * data,
        माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    माप_प्रकार आदान_size = आकार(rpc_tensor) + 4*आकार(uint64_t) + size*n_copies;
    uint8_t * आदान = नवीन uint8_t[आदान_size]();
    uint8_t * dest = आदान;
    स_नकल(dest, &rpc_tensor, आकार(rpc_tensor));
    dest += आकार(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    स_नकल(dest, header, आकार(header));
    dest += आकार(header);
    साठी (माप_प्रकार i = 0; i < n_copies; i++) {
        स_नकल(dest + i*size, (स्थिरांक अक्षर *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> आदान_ptr(आदान, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, आदान_ptr, आदान_size);
}

स्थिर शून्य ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, स्थिरांक ggml_tensor * tensor, शून्य * data,
        माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    स्वतः request = std::make_shared<rpc_संदेश_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    जर (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, आकार(*request), data, size*n_copies);
    } अन्यथा {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, आकार(*request), packed.data(), packed.size());
        साठी (माप_प्रकार i = 0; i < n_copies; i++) {
            स_नकल((अक्षर *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

स्थिर शून्य ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, स्थिरांक ggml_tensor * tensor, शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    स्वतः request = std::make_shared<rpc_संदेश_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, आकार(*request), data, size);
}

स्थिर तार्किक ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, स्थिरांक ggml_tensor * src, ggml_tensor * dst) {
    जर (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        जर (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            परत असत्य;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        स्वतः request = std::make_shared<rpc_संदेश_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_संदेश_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, आकार(*request), &response, आकार(response));
        परत response.result;
    }
    परत असत्य;
}

स्थिर शून्य ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    स्वतः request = std::make_shared<rpc_संदेश_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, आकार(*request));
}

स्थिर ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
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
    /* .reset           = */ रिक्त,
};

स्थिर स्थिरांक अक्षर * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    परत buft_ctx->name.c_str();
}

स्थिर ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    स्वतः request = std::make_shared<rpc_संदेश_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_संदेश_alloc_buffer_rsp response;

    स्वतः dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, आकार(*request), &response, आकार(response));
    जर (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            नवीन ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        परत buffer;
    } अन्यथा {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        परत nullptr;
    }
}

स्थिर माप_प्रकार get_alignment(स्थिरांक std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    स्वतः request = std::make_shared<rpc_संदेश_get_alignment_req>();
    request->device = device;
    rpc_संदेश_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, आकार(*request), &response, आकार(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    परत response.alignment;
}

स्थिर माप_प्रकार ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    परत buft_ctx->alignment;
}

स्थिर माप_प्रकार get_max_size(स्थिरांक std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    स्वतः request = std::make_shared<rpc_संदेश_get_max_size_req>();
    request->device = device;
    rpc_संदेश_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, आकार(*request), &response, आकार(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    परत response.max_size;
}

स्थिर माप_प्रकार ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    परत buft_ctx->max_size;
}

स्थिर माप_प्रकार ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, स्थिरांक ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    तार्किक rpc_get = असत्य;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    जर (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        स्थिरांक माप_प्रकार min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        संरचना alloc_size_cache_किल्ली {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / आकार(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_किल्ली किल्ली = {};
        किल्ली.device = buft_ctx->device;
        किल्ली.type = tensor->type;
        किल्ली.op = tensor->op;
        स_नकल(किल्ली.op_params, tensor->op_params, आकार(किल्ली.op_params));
        साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
            किल्ली.ne[i] = (uint32_t)tensor->ne[i];
            किल्ली.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((स्थिरांक uint8_t *)&किल्ली, आकार(किल्ली));
        cache_hash = fnv_hash((स्थिरांक uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        स्थिर std::mutex cache_mutex;
        स्थिर std::unordered_map<uint64_t, माप_प्रकार> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            स्वतः it = cache.find(cache_hash);
            जर (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                परत std::max<माप_प्रकार>(it->second, min_size);
            }
        }

        स्वतः request = std::make_shared<rpc_संदेश_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_संदेश_get_alloc_size_rsp response;
        स्वतः dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, आकार(*request), &response, आकार(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        परत std::max<माप_प्रकार>(response.alloc_size, min_size);
    }

    परत ggml_nbytes(tensor);
}

स्थिर ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};

स्थिर स्थिरांक अक्षर * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    परत rpc_ctx->name.c_str();
}

स्थिर शून्य ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

स्थिर शून्य ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    जर (rpc_use_hash_cache(tensor, size)) {
        स्वतः request = std::make_shared<rpc_संदेश_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((स्थिरांक uint8_t*)data, size);
        rpc_संदेश_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, आकार(*request), &response, आकार(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        जर (response.result) {
            // the server has the same data, no need to send it
            परत;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    माप_प्रकार आदान_size;
    स्वतः आदान = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, आदान_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, आदान, आदान_size);
}

स्थिर शून्य ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, स्थिरांक ggml_tensor * tensor, शून्य * data, माप_प्रकार offset, माप_प्रकार size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    स्वतः request = std::make_shared<rpc_संदेश_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, आकार(*request), data, size);
}

स्थिर शून्य ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

स्थिर शून्य add_tensor(ggml_tensor * tensor, स्थिरांक ggml_cgraph * cgraph, स्थिरांक rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    जर (tensor == nullptr) {
        परत;
    }
    जर (visited.find(tensor) != visited.end()) {
        परत;
    }
    visited.insert(tensor);
    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    स्थिरांक माप_प्रकार hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    जर (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_मोजणी = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

स्थिर uint8_t * serialize_graph(uint32_t device, स्थिरांक ggml_cgraph * cgraph, स्थिरांक rpc_dispatcher * dispatcher, माप_प्रकार * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    साठी (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*आकार(uint32_t) + आकार(uint64_t) + n_nodes * आकार(uint64_t) + आकार(uint32_t) + n_tensors * आकार(rpc_tensor);
    uint8_t * output = नवीन uint8_t[*output_size]();
    uint8_t * dest = output;
    स_नकल(dest, &device, आकार(device));
    dest += आकार(device);
    स_नकल(dest, &cgraph->uid, आकार(cgraph->uid));
    dest += आकार(cgraph->uid);
    स_नकल(dest, &n_nodes, आकार(n_nodes));
    dest += आकार(n_nodes);
    साठी (uint32_t i = 0; i < n_nodes; i++) {
        स_नकल(dest + i * आकार(uint64_t), &cgraph->nodes[i], आकार(uint64_t));
    }
    dest += n_nodes * आकार(uint64_t);
    स_नकल(dest, &n_tensors, आकार(n_tensors));
    dest += आकार(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    स_नकल(out_tensors, tensors.data(), n_tensors * आकार(rpc_tensor));
    परत output;
}

शून्य rpc_dispatcher::graph_compute(uint32_t device, स्थिरांक ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    स्वतः & device_graph_uids = graph_uids[device];
    तार्किक reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    जर (reuse) {
        स्वतः request = std::make_shared<rpc_संदेश_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, आकार(*request));
    } अन्यथा {
        जर (cgraph->uid != 0) {
            जर (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        माप_प्रकार आदान_size = 0;
        uint8_t * आदान = serialize_graph(device, cgraph, हे, &आदान_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(आदान_size).c_str());
        std::shared_ptr<uint8_t> आदान_ptr(आदान, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, आदान_ptr, आदान_size);
    }
}

स्थिर गणना ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    परत GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

स्थिर शून्य ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

स्थिर शून्य ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, स्थिरांक शून्य * data,
        माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

स्थिर शून्य ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, स्थिरांक ggml_tensor * tensor, शून्य * data,
        माप_प्रकार offset, माप_प्रकार size, माप_प्रकार n_copies, माप_प्रकार stride_tensor, माप_प्रकार stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

स्थिर ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ रिक्त,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ रिक्त,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(स्थिरांक अक्षर * endpoint, uint32_t device) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    स्थिर std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    स्वतः it = buft_map.find(buft_name);
    जर (it != buft_map.end()) {
        परत it->second;
    }
    स्वतः dispatcher = get_dispatcher(endpoint);
    माप_प्रकार alignment = get_alignment(dispatcher, device);
    माप_प्रकार max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = नवीन ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    स्वतः reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = नवीन ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    परत buft;
}

ggml_backend_t ggml_backend_rpc_init(स्थिरांक अक्षर * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    स्वतः dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = नवीन ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    स्वतः reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = नवीन ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    परत backend;
}

तार्किक ggml_backend_is_rpc(ggml_backend_t backend) {
    परत backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

शून्य ggml_backend_rpc_get_device_memory(स्थिरांक अक्षर * endpoint, uint32_t device, माप_प्रकार * मुक्त, माप_प्रकार * total) {
    स्वतः dispatcher = get_dispatcher(endpoint);
    स्वतः request = std::make_shared<rpc_संदेश_get_device_memory_req>();
    request->device = device;
    rpc_संदेश_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, आकार(*request), &response, आकार(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *मुक्त = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

वर्ग rpc_server {
सार्वजनिक:
    rpc_server(std::vector<ggml_backend_t> all_backends, स्थिरांक अक्षर * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    शून्य hello(rpc_संदेश_hello_rsp & response);
    तार्किक alloc_buffer(स्थिरांक rpc_संदेश_alloc_buffer_req & request, rpc_संदेश_alloc_buffer_rsp & response);
    तार्किक get_alignment(स्थिरांक rpc_संदेश_get_alignment_req & request, rpc_संदेश_get_alignment_rsp & response);
    तार्किक get_max_size(स्थिरांक rpc_संदेश_get_max_size_req & request, rpc_संदेश_get_max_size_rsp & response);
    तार्किक buffer_get_base(स्थिरांक rpc_संदेश_buffer_get_base_req & request, rpc_संदेश_buffer_get_base_rsp & response);
    तार्किक free_buffer(स्थिरांक rpc_संदेश_free_buffer_req & request);
    तार्किक buffer_clear(स्थिरांक rpc_संदेश_buffer_clear_req & request);
    तार्किक memset_tensor(स्थिरांक rpc_संदेश_memset_tensor_req & request);
    तार्किक set_tensor(स्थिरांक std::vector<uint8_t> & आदान);
    तार्किक set_tensor_2d(स्थिरांक std::vector<uint8_t> & आदान);
    तार्किक set_tensor_hash(स्थिरांक rpc_संदेश_set_tensor_hash_req & request, rpc_संदेश_set_tensor_hash_rsp & response);
    तार्किक get_tensor(स्थिरांक rpc_संदेश_get_tensor_req & request, std::vector<uint8_t> & response);
    तार्किक get_tensor_2d(स्थिरांक rpc_संदेश_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    तार्किक copy_tensor(स्थिरांक rpc_संदेश_copy_tensor_req & request, rpc_संदेश_copy_tensor_rsp & response);
    तार्किक graph_compute(स्थिरांक std::vector<uint8_t> & आदान);
    तार्किक graph_recompute(स्थिरांक rpc_संदेश_graph_recompute_req & request);
    तार्किक comm_init(स्थिरांक rpc_संदेश_comm_init_req & request, rpc_संदेश_comm_init_rsp & response);
    तार्किक comm_allreduce(स्थिरांक rpc_संदेश_comm_allreduce_req & request);
    तार्किक comm_free(स्थिरांक rpc_संदेश_comm_free_req & request);
    तार्किक init_tensor(स्थिरांक rpc_संदेश_init_tensor_req & request);
    तार्किक get_alloc_size(स्थिरांक rpc_संदेश_get_alloc_size_req & request, rpc_संदेश_get_alloc_size_rsp & response);
    तार्किक get_device_memory(स्थिरांक rpc_संदेश_get_device_memory_req & request, rpc_संदेश_get_device_memory_rsp & response);

    संरचना stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

खाजगी:
    शून्य sync_all_backends();
    तार्किक get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(संरचना ggml_context * ctx, स्थिरांक rpc_tensor * tensor);
    ggml_tensor * create_गाठ(uint64_t id,
                              संरचना ggml_context * ctx,
                              स्थिरांक std::unordered_map<uint64_t, स्थिरांक rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, संरचना ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    संरचना comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        माप_प्रकार                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    स्थिरांक अक्षर * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

शून्य rpc_server::hello(rpc_संदेश_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

तार्किक rpc_server::get_alloc_size(स्थिरांक rpc_संदेश_get_alloc_size_req & request, rpc_संदेश_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    जर (dev_id >= backends.size()) {
        परत असत्य;
    }
    ggml_backend_buffer_type_t buft;
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    जर (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        परत असत्य;
    }
    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        जर (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (शून्य*)tensor->buffer, tensor->data);
    जर (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } अन्यथा {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    परत सत्य;
}

तार्किक rpc_server::alloc_buffer(स्थिरांक rpc_संदेश_alloc_buffer_req & request, rpc_संदेश_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    जर (dev_id >= backends.size()) {
        परत असत्य;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    जर (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } अन्यथा {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    परत सत्य;
}

तार्किक rpc_server::get_alignment(स्थिरांक rpc_संदेश_get_alignment_req & request, rpc_संदेश_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    जर (dev_id >= backends.size()) {
        परत असत्य;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    माप_प्रकार alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    परत सत्य;
}

तार्किक rpc_server::get_max_size(स्थिरांक rpc_संदेश_get_max_size_req & request, rpc_संदेश_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    जर (dev_id >= backends.size()) {
        परत असत्य;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    माप_प्रकार max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    परत सत्य;
}

तार्किक rpc_server::buffer_get_base(स्थिरांक rpc_संदेश_buffer_get_base_req & request, rpc_संदेश_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    जर (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        परत असत्य;
    }
    शून्य * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    परत सत्य;
}

तार्किक rpc_server::free_buffer(स्थिरांक rpc_संदेश_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    जर (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        परत असत्य;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    साठी (स्वतः & sgs : stored_graphs) {
        साठी (स्वतः & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    परत सत्य;
}

तार्किक rpc_server::buffer_clear(स्थिरांक rpc_संदेश_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    जर (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        परत असत्य;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    परत सत्य;
}

तार्किक rpc_server::memset_tensor(स्थिरांक rpc_संदेश_memset_tensor_req & request) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    जर (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        परत असत्य;
    }

    स्थिरांक uint64_t tensor_size = ggml_nbytes(tensor);
    जर (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        परत असत्य;
    }

    स्थिरांक uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    स्थिरांक uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    जर (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        परत असत्य;
    }
    स्थिरांक uint64_t data_offset = request.tensor.data - buffer_start;
    जर (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        परत असत्य;
    }
    जर (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        परत असत्य;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    परत सत्य;
}

ggml_tensor * rpc_server::deserialize_tensor(संरचना ggml_context * ctx, स्थिरांक rpc_tensor * tensor) {
    // Validate tensor type before using it
    जर (tensor->type >= GGML_TYPE_मोजणी) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        परत nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    जर (ggml_blck_size((गणना ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        परत nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    जर (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        परत nullptr;
    }

    साठी (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    जर (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    जर (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        जर (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            परत nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    साठी (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / आकार(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<शून्य *>(tensor->data);
    ggml_set_name(result, tensor->name);
    परत result;
}


तार्किक rpc_server::set_tensor(स्थिरांक std::vector<uint8_t> & आदान) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    स्थिरांक माप_प्रकार header_size = आकार(rpc_tensor) + आकार(cache_flag) + आकार(offset);
    जर (आदान.size() < header_size) {
        परत असत्य;
    }
    स्थिरांक rpc_tensor * in_tensor = (स्थिरांक rpc_tensor *)आदान.data();
    स_नकल(&cache_flag, आदान.data() + आकार(rpc_tensor), आकार(cache_flag));
    स_नकल(&offset,     आदान.data() + आकार(rpc_tensor) + आकार(cache_flag), आकार(offset));
    स्थिरांक माप_प्रकार size = आदान.size() - header_size;

    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    जर (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        परत असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        स्थिरांक माप_प्रकार p0 = (माप_प्रकार) ggml_backend_buffer_get_base(tensor->buffer);
        स्थिरांक माप_प्रकार p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        जर (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            परत असत्य;
        }
    }

    स्थिरांक शून्य * data = आदान.data() + header_size;
    जर (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((स्थिरांक uint8_t*)data, size);
        अक्षर hash_str[17];
        snprintf(hash_str, आकार(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.लिख((स्थिरांक अक्षर *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    परत सत्य;
}

तार्किक rpc_server::set_tensor_2d(स्थिरांक std::vector<uint8_t> & आदान) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    जर (आदान.size() < आकार(rpc_tensor) + 4*आकार(uint64_t)) {
        परत असत्य;
    }
    स्थिरांक rpc_tensor * in_tensor = (स्थिरांक rpc_tensor *)आदान.data();
    uint64_t header[4];
    स_नकल(header, आदान.data() + आकार(rpc_tensor), आकार(header));
    स्थिरांक uint64_t offset   = header[0];
    स्थिरांक uint64_t size     = header[1];
    स्थिरांक uint64_t n_copies = header[2];
    स्थिरांक uint64_t stride   = header[3];

    स्थिरांक uint64_t data_size = आदान.size() - आकार(rpc_tensor) - 4*आकार(uint64_t);
    जर (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        परत असत्य;
    }

    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    जर (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        परत असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        जर (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            परत असत्य;
        }
        स्थिरांक uint64_t span = (n_copies - 1)*stride + size;
        स्थिरांक uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        स्थिरांक uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        जर (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            परत असत्य;
        }
        जर (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            परत असत्य;
        }
    }

    स्थिरांक शून्य * data = आदान.data() + आकार(rpc_tensor) + 4*आकार(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    परत सत्य;
}

तार्किक rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    जर (!cache_dir) {
        परत असत्य;
    }
    अक्षर hash_str[17];
    snprintf(hash_str, आकार(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    जर (!fs::exists(cache_file, ec)) {
        परत असत्य;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    माप_प्रकार size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.पठ((अक्षर *)data.data(), size);
    परत सत्य;
}

तार्किक rpc_server::set_tensor_hash(स्थिरांक rpc_संदेश_set_tensor_hash_req & request, rpc_संदेश_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    जर (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        परत असत्य;
    }

    std::vector<uint8_t> cached_file;
    जर (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        परत सत्य;
    }
    माप_प्रकार size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        स्थिरांक माप_प्रकार p0 = (माप_प्रकार) ggml_backend_buffer_get_base(tensor->buffer);
        स्थिरांक माप_प्रकार p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        जर (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            परत असत्य;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    परत सत्य;
}

तार्किक rpc_server::init_tensor(स्थिरांक rpc_संदेश_init_tensor_req & request) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    जर (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        परत असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (शून्य*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    जर (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } अन्यथा {
        जर (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    जर (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        परत असत्य;
    }

    परत सत्य;
}

तार्किक rpc_server::get_tensor(स्थिरांक rpc_संदेश_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    जर (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        परत असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        स्थिरांक माप_प्रकार p0 = (माप_प्रकार) ggml_backend_buffer_get_base(tensor->buffer);
        स्थिरांक माप_प्रकार p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        जर (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                परत असत्य;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    परत सत्य;
}

तार्किक rpc_server::get_tensor_2d(स्थिरांक rpc_संदेश_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    जर (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        परत असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        जर (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            परत असत्य;
        }
        जर (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            परत असत्य;
        }
        स्थिरांक uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        स्थिरांक uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        स्थिरांक uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        जर (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            परत असत्य;
        }
        जर (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            परत असत्य;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    परत सत्य;
}

तार्किक rpc_server::copy_tensor(स्थिरांक rpc_संदेश_copy_tensor_req & request, rpc_संदेश_copy_tensor_rsp & response) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    जर (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        परत असत्य;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    जर (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        परत असत्य;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    परत सत्य;
}

ggml_tensor * rpc_server::create_गाठ(uint64_t id,
                                      संरचना ggml_context * ctx,
                                      स्थिरांक std::unordered_map<uint64_t, स्थिरांक rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, संरचना ggml_tensor*> & tensor_map) {
    जर (tensor_map.find(id) != tensor_map.end()) {
        परत tensor_map[id];
    }
    // Safely find the tensor pointer
    स्वतः it_ptr = tensor_ptrs.find(id);
    जर (it_ptr == tensor_ptrs.end()) {
        परत nullptr;
    }
    स्थिरांक rpc_tensor * tensor = it_ptr->second;

    संरचना ggml_tensor * result = deserialize_tensor(ctx, tensor);
    जर (result == nullptr) {
        परत nullptr;
    }
    जर (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        परत nullptr;
    }
    tensor_map[id] = result;
    साठी (पूर्णांक i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        जर (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } अन्यथा {
            result->src[i] = create_गाठ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            जर (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                परत nullptr;
            }
        }
    }

    // Handle view_src similarly
    जर (tensor->view_src == 0) {
        result->view_src = nullptr;
    } अन्यथा {
        result->view_src = create_गाठ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        जर (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            परत nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    परत result;
}

तार्किक rpc_server::graph_compute(स्थिरांक std::vector<uint8_t> & आदान) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    जर (आदान.size() < 2*आकार(uint32_t) + आकार(uint64_t)) {
        परत असत्य;
    }
    स्थिरांक uint8_t * src = आदान.data();
    uint32_t device;
    स_नकल(&device, src, आकार(device));
    src += आकार(device);
    जर (device >= backends.size()) {
        परत असत्य;
    }
    uint64_t uid;
    स_नकल(&uid, src, आकार(uid));
    src += आकार(uid);
    uint32_t n_nodes;
    स_नकल(&n_nodes, src, आकार(n_nodes));
    src += आकार(n_nodes);
    जर (आदान.size() < 2*आकार(uint32_t) + आकार(uint64_t) + n_nodes*आकार(uint64_t) + आकार(uint32_t)) {
        परत असत्य;
    }
    स्थिरांक uint64_t * nodes = (स्थिरांक uint64_t *)src;
    src += n_nodes*आकार(uint64_t);
    uint32_t n_tensors;
    स_नकल(&n_tensors, src, आकार(n_tensors));
    src += आकार(n_tensors);
    जर (आदान.size() < 2*आकार(uint32_t) + आकार(uint64_t) + n_nodes*आकार(uint64_t) + आकार(uint32_t) + n_tensors*आकार(rpc_tensor)) {
        परत असत्य;
    }
    स्थिरांक rpc_tensor * tensors = (स्थिरांक rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    जर (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    माप_प्रकार buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, असत्य);
    जर (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    संरचना ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    संरचना ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, असत्य);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, स्थिरांक rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    साठी (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    स्थिरांक स्वतः t_deserialize = std::chrono::steady_clock::now();
    साठी (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        स_नकल(&id, &nodes[i], आकार(id));
        graph->nodes[i] = create_गाठ(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        जर (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            परत असत्य;
        }
        जर (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            स्थिरांक माप_प्रकार hash_pos = ggml_hash_घाला(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_मोजणी;
        }
    }
    स्थिरांक std::chrono::duration<द्विगुण, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    परत सत्य;
}

तार्किक rpc_server::graph_recompute(स्थिरांक rpc_संदेश_graph_recompute_req & request) {
    uint32_t device = request.device;
    जर (device >= backends.size()) {
        परत असत्य;
    }
    स्वतः it = stored_graphs[device].find(request.uid);
    जर (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        परत असत्य;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    परत सत्य;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
शून्य rpc_server::sync_all_backends() {
    साठी (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
तार्किक rpc_server::comm_init(स्थिरांक rpc_संदेश_comm_init_req & request, rpc_संदेश_comm_init_rsp & response) {
    response.ok = 0;
    जर (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        परत सत्य;
    }
    comm_state & state = comm_states[request.device];
    जर (state.peer != nullptr) {
        response.ok = 1;
        परत सत्य;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    जर (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        जर (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            परत सत्य;
        }
        state.peer = srv->accept();
        जर (state.peer == nullptr) {
            परत सत्य;
        }
        जर (!state.peer->recv_data(remote_caps, आकार(remote_caps))) {
            state.peer = nullptr;
            परत सत्य;
        }
        state.peer->get_caps(local_caps);
        जर (!state.peer->send_data(local_caps, आकार(local_caps))) {
            state.peer = nullptr;
            परत सत्य;
        }
        state.peer->update_caps(remote_caps);
    } अन्यथा {
        स्थिरांक std::string host(request.host, strnlen(request.host, आकार(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        साठी (पूर्णांक i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            जर (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        जर (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            परत सत्य;
        }
        state.peer->get_caps(local_caps);
        जर (!state.peer->send_data(local_caps, आकार(local_caps)) ||
            !state.peer->recv_data(remote_caps, आकार(remote_caps))) {
            state.peer = nullptr;
            परत सत्य;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    परत सत्य;
}

तार्किक rpc_server::comm_allreduce(स्थिरांक rpc_संदेश_comm_allreduce_req & request) {
    जर (request.device >= backends.size()) {
        परत असत्य;
    }
    comm_state & state = comm_states[request.device];
    जर (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        परत असत्य;
    }
    ggml_backend_t backend = backends[request.device];

    माप_प्रकार ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, असत्य);
    संरचना ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    जर (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        परत असत्य;
    }
    स्थिरांक माप_प्रकार  nbytes = ggml_nbytes(t_dst);
    स्थिरांक int64_t ne     = ggml_nelements(t_dst);
    जर (nbytes == 0) {
        परत सत्य;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    स्थिरांक तार्किक   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    स्थिरांक माप_प्रकार wire_bytes = wire_bf16 ? (माप_प्रकार) ne*2 : nbytes;
    स्थिरांक माप_प्रकार need       = wire_bf16 ? 2*nbytes : nbytes;
    जर (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    अक्षर * scratch_base = (अक्षर *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    स्वतः new_scratch_tensor = [&](ggml_type type, माप_प्रकार offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        परत t;
    };
    स्वतः new_cpy_गाठ = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        परत t;
    };
    स्वतः compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, असत्य);
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
    जर (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_गाठ(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } अन्यथा {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    जर (state.rank == 0) {
        जर (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            परत असत्य;
        }
    } अन्यथा {
        जर (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            परत असत्य;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (माप_प्रकार) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    जर (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_गाठ(t_wire_recv, t_peer);
    } अन्यथा {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    जर (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } अन्यथा {
        compute_nodes(t_red, nullptr);
    }
    परत सत्य;
}

तार्किक rpc_server::comm_free(स्थिरांक rpc_संदेश_comm_free_req & request) {
    जर (request.device >= backends.size()) {
        परत असत्य;
    }
    comm_states[request.device] = comm_state();
    परत सत्य;
}

तार्किक rpc_server::get_device_memory(स्थिरांक rpc_संदेश_get_device_memory_req & request, rpc_संदेश_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    जर (dev_id >= backends.size()) {
        परत असत्य;
    }
    माप_प्रकार मुक्त, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &मुक्त, &total);
    response.free_mem = मुक्त;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    परत सत्य;
}

rpc_server::~rpc_server() {
    साठी (स्वतः buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

स्थिर शून्य rpc_serve_client(स्थिरांक std::vector<ggml_backend_t> & backends, स्थिरांक अक्षर * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    जर (!sock->recv_data(&cmd, 1)) {
        परत;
    }
    जर (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        परत;
    }

    // Read input_size and validate protocol version
    uint64_t hello_आदान_size;
    जर (!sock->recv_data(&hello_आदान_size, आकार(hello_आदान_size))) {
        परत;
    }

    जर (hello_आदान_size != आकार(rpc_संदेश_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (माप_प्रकार)hello_आदान_size, आकार(rpc_संदेश_hello_req), RPC_PROTO_MAJOR_VERSION);
        परत;
    }

    rpc_संदेश_hello_req req = {};
    जर (!sock->recv_data(&req, आकार(req))) {
        परत;
    }

    rpc_संदेश_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    जर (!send_संदेश(sock, &rsp, आकार(rsp))) {
        परत;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    पर्यंत (सत्य) {
        जर (!sock->recv_data(&cmd, 1)) {
            थांबा;
        }
        जर (cmd >= RPC_CMD_मोजणी) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            थांबा;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((गणना rpc_cmd)cmd));
        निवड (cmd) {
            स्थिती RPC_CMD_HELLO: {
                // HELLO command is handled above
                परत;
            }
            स्थिती RPC_CMD_DEVICE_मोजणी: {
                जर (!recv_संदेश(sock, nullptr, 0)) {
                    परत;
                }
                rpc_संदेश_device_मोजणी_rsp response;
                response.device_मोजणी = backends.size();
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_ALLOC_BUFFER: {
                rpc_संदेश_alloc_buffer_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_alloc_buffer_rsp response;
                जर (!server.alloc_buffer(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_GET_ALLOC_SIZE: {
                rpc_संदेश_get_alloc_size_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_get_alloc_size_rsp response;
                जर (!server.get_alloc_size(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_GET_ALIGNMENT: {
                rpc_संदेश_get_alignment_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_get_alignment_rsp response;
                जर (!server.get_alignment(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_GET_MAX_SIZE: {
                rpc_संदेश_get_max_size_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_get_max_size_rsp response;
                जर (!server.get_max_size(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_BUFFER_GET_BASE: {
                rpc_संदेश_buffer_get_base_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_buffer_get_base_rsp response;
                जर (!server.buffer_get_base(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_FREE_BUFFER: {
                rpc_संदेश_free_buffer_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                जर (!server.free_buffer(request)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_BUFFER_CLEAR: {
                rpc_संदेश_buffer_clear_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                जर (!server.buffer_clear(request)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_MEMSET_TENSOR: {
                rpc_संदेश_memset_tensor_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                जर (!server.memset_tensor(request)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> आदान;
                जर (!recv_संदेश(sock, आदान)) {
                    परत;
                }
                जर (!server.set_tensor(आदान)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> आदान;
                जर (!recv_संदेश(sock, आदान)) {
                    परत;
                }
                जर (!server.set_tensor_2d(आदान)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_GET_TENSOR_2D: {
                rpc_संदेश_get_tensor_2d_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                std::vector<uint8_t> response;
                जर (!server.get_tensor_2d(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, response.data(), response.size())) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_SET_TENSOR_HASH: {
                rpc_संदेश_set_tensor_hash_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_set_tensor_hash_rsp response;
                जर (!server.set_tensor_hash(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_INIT_TENSOR: {
                rpc_संदेश_init_tensor_req request;
                जर (!recv_संदेश(sock, &request,आकार(request))) {
                    परत;
                }
                जर (!server.init_tensor(request)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_GET_TENSOR: {
                rpc_संदेश_get_tensor_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                std::vector<uint8_t> response;
                जर (!server.get_tensor(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, response.data(), response.size())) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_COPY_TENSOR: {
                rpc_संदेश_copy_tensor_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_copy_tensor_rsp response;
                जर (!server.copy_tensor(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> आदान;
                जर (!recv_संदेश(sock, आदान)) {
                    परत;
                }
                जर (!server.graph_compute(आदान)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_संदेश_graph_recompute_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                जर (!server.graph_recompute(request)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_COMM_INIT: {
                rpc_संदेश_comm_init_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_comm_init_rsp response;
                जर (!server.comm_init(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_COMM_ALLREDUCE: {
                rpc_संदेश_comm_allreduce_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                जर (!server.comm_allreduce(request)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_COMM_FREE: {
                rpc_संदेश_comm_free_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                जर (!server.comm_free(request)) {
                    परत;
                }
                थांबा;
            }
            स्थिती RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_संदेश_get_device_memory_req request;
                जर (!recv_संदेश(sock, &request, आकार(request))) {
                    परत;
                }
                rpc_संदेश_get_device_memory_rsp response;
                जर (!server.get_device_memory(request, response)) {
                    परत;
                }
                जर (!send_संदेश(sock, &response, आकार(response))) {
                    परत;
                }
                थांबा;
            }
            पूर्वनिर्धारित: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((गणना rpc_cmd)cmd));
                परत;
            }
        }
    }
}

शून्य ggml_backend_rpc_start_server(स्थिरांक अक्षर * endpoint, स्थिरांक अक्षर * cache_dir,
                                   माप_प्रकार n_threads, माप_प्रकार n_devices, ggml_backend_dev_t * devices) {
    जर (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        परत;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    साठी (माप_प्रकार i = 0; i < n_devices; i++) {
        स्वतः dev = devices[i];
        माप_प्रकार मुक्त, total;
        ggml_backend_dev_memory(dev, &मुक्त, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, मुक्त / 1024 / 1024);
        स्वतः backend = ggml_backend_dev_init(dev, nullptr);
        जर (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            परत;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        जर (reg) {
            स्वतः ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            जर (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    पूर्णांक port;
    जर (!parse_endpoint(endpoint, host, port)) {
        परत;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#अन्यथा
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    जर (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        परत;
    }
    स्वतः server_socket = socket_t::create_server(host.c_str(), port);
    जर (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        परत;
    }
    पर्यंत (सत्य) {
        स्वतः client_socket = server_socket->accept();
        जर (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            परत;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    साठी (स्वतः backend : backends) {
        ggml_backend_free(backend);
    }
}

स्थिर स्थिरांक अक्षर * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    परत ctx->name.c_str();
}

स्थिर स्थिरांक अक्षर * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    परत ctx->description.c_str();
}

स्थिर शून्य ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, मुक्त, total);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    परत GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

स्थिर शून्य ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ सत्य,
        /* .host_buffer           = */ असत्य,
        /* .buffer_from_host_ptr  = */ असत्य,
        /* .events                = */ सत्य,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, स्थिरांक अक्षर * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    परत ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    परत ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

स्थिर तार्किक ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, स्थिरांक संरचना ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    परत सत्य;
}

स्थिर तार्किक ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    जर (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        परत असत्य;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    परत buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

स्थिर ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वतः dispatcher = get_dispatcher(ctx->endpoint);
    परत dispatcher->event_new(dev);
}

स्थिर शून्य ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वतः dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

स्थिर शून्य ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वतः dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

स्थिर स्थिरांक संरचना ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ रिक्त,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ रिक्त,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

संरचना ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

स्थिर स्थिरांक अक्षर * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    परत ctx ? ctx->name.c_str() : "RPC";
}

स्थिर माप_प्रकार ggml_backend_rpc_reg_get_device_मोजणी(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    परत ctx ? ctx->devices.size() : 0;
}

स्थिर ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    जर (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } अन्यथा {
        GGML_ASSERT(index < ctx->devices.size());
        परत ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
संरचना ggml_backend_rpc_comm_context {
    संरचना rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

स्थिर शून्य ggml_backend_rpc_comm_free(शून्य * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    जर (comm_ctx == nullptr) {
        परत;
    }
    साठी (स्थिरांक स्वतः & rank : comm_ctx->ranks) {
        स्वतः request = std::make_shared<rpc_संदेश_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, आकार(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

स्थिर शून्य * ggml_backend_rpc_comm_init(ggml_backend_t * backends, माप_प्रकार n_backends) {
    जर (n_backends != 2 || std::दो_पर्या("GGML_RPC_NO_COMM") != nullptr) {
        परत nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        जर (!ggml_backend_is_rpc(backends[i])) {
            परत nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        साठी (स्थिरांक स्वतः & rank : ranks) {
            जर (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                परत nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    पूर्णांक port0;
    जर (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        परत nullptr;
    }
    स्थिरांक uint32_t comm_port = (uint32_t) port0 + 1000;
    जर (host0.size() >= 64) {
        परत nullptr;
    }

    साठी (स्थिरांक स्वतः & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_संदेश_comm_init_rsp> responses(n_backends);
    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        स्वतः request = std::make_shared<rpc_संदेश_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        जर (i > 0) {
            स_नकल(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, आकार(*request), &responses[i], आकार(responses[i]));
    }
    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    तार्किक ok = सत्य;
    साठी (माप_प्रकार i = 0; i < n_backends; i++) {
        जर (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = असत्य;
        }
    }
    जर (!ok) {
        साठी (स्थिरांक स्वतः & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        परत nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    परत नवीन ggml_backend_rpc_comm_context{std::move(ranks)};
}

स्थिर तार्किक ggml_backend_rpc_comm_allreduce_tensor(शून्य * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    जर (comm_ctx == nullptr) {
        परत असत्य;
    }
    स्थिरांक माप_प्रकार n_ranks = comm_ctx->ranks.size();
    स्थिरांक int64_t ne = ggml_nelements(tensors[0]);
    जर (ne == 0) {
        परत सत्य;
    }
    साठी (माप_प्रकार i = 0; i < n_ranks; i++) {
        जर (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            परत असत्य;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        जर ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            परत असत्य;
        }
    }
    साठी (माप_प्रकार i = 0; i < n_ranks; i++) {
        स्वतः request = std::make_shared<rpc_संदेश_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, आकार(*request));
    }
    परत सत्य;
}

स्थिर शून्य * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, स्थिरांक अक्षर * name) {
    जर (std::म_भेद(name, "ggml_backend_rpc_add_server") == 0) {
        परत (शून्य *)ggml_backend_rpc_add_server;
    }
    जर (std::म_भेद(name, "ggml_backend_rpc_start_server") == 0) {
        परत (शून्य *)ggml_backend_rpc_start_server;
    }
    जर (std::म_भेद(name, "ggml_backend_comm_init") == 0) {
        परत (शून्य *)ggml_backend_rpc_comm_init;
    }
    जर (std::म_भेद(name, "ggml_backend_comm_free") == 0) {
        परत (शून्य *)ggml_backend_rpc_comm_free;
    }
    जर (std::म_भेद(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        परत (शून्य *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    परत रिक्त;

    GGML_UNUSED(reg);
}

स्थिर स्थिरांक संरचना ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_मोजणी,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(शून्य) {
    स्थिर संरचना ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ रिक्त,
    };

    परत &ggml_backend_rpc_reg;
}

स्थिर uint32_t ggml_backend_rpc_get_device_मोजणी(स्थिरांक अक्षर * endpoint) {
    स्वतः dispatcher = get_dispatcher(endpoint);
    rpc_संदेश_device_मोजणी_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_मोजणी, nullptr, 0, &response, आकार(response));
    परत response.device_मोजणी;
}

स्थिर स्थिरांक ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_मोजणी,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(स्थिरांक अक्षर * endpoint) {
    स्थिर std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    स्थिर std::mutex mutex;
    स्थिर uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    जर (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        परत reg_map[endpoint];
    }
    uint32_t dev_मोजणी = ggml_backend_rpc_get_device_मोजणी(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_मोजणी);
    जर (dev_मोजणी == 0) {
        परत nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = नवीन ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    साठी (uint32_t ind = 0; ind < dev_मोजणी; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = नवीन ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = नवीन ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = नवीन ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    परत reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

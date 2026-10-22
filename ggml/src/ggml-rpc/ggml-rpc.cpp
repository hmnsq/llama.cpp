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
#परिभाषय RPC_स्थितिः_ASSERT(x) यदि (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#निर्देशः pack(प्रक्षिप, 1)
// ggml_tensor is serialized into rpc_tensor
संरचना rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t बफर;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / परिमाण(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t दत्त;
    वर्ण name[GGML_MAX_NAME];

    int32_t use_गणनम्;
};

static_assert(परिमाण(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
गणना rpc_cmd {
    RPC_CMD_ALLOC_बफर = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_आकार,
    RPC_CMD_बफर_GET_आधारः,
    RPC_CMD_FREE_बफर,
    RPC_CMD_बफर_CLEAR,
    RPC_CMD_समुच्चयः_TENSOR,
    RPC_CMD_समुच्चयः_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_स्मृतिः,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_आकार,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_गणनम्,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_समुच्चयः_TENSOR_2_D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_गणनम्,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

स्थिर ध्रुव वर्ण * rpc_cmd_name(गणना rpc_cmd cmd) {
    विकल्प (cmd) {
        पक्ष RPC_CMD_ALLOC_बफर:      प्रत्यागम "ALLOC_BUFFER";
        पक्ष RPC_CMD_GET_ALIGNMENT:     प्रत्यागम "GET_ALIGNMENT";
        पक्ष RPC_CMD_GET_MAX_आकार:      प्रत्यागम "GET_MAX_SIZE";
        पक्ष RPC_CMD_बफर_GET_आधारः:   प्रत्यागम "BUFFER_GET_BASE";
        पक्ष RPC_CMD_FREE_बफर:       प्रत्यागम "FREE_BUFFER";
        पक्ष RPC_CMD_बफर_CLEAR:      प्रत्यागम "BUFFER_CLEAR";
        पक्ष RPC_CMD_समुच्चयः_TENSOR:        प्रत्यागम "SET_TENSOR";
        पक्ष RPC_CMD_समुच्चयः_TENSOR_HASH:   प्रत्यागम "SET_TENSOR_HASH";
        पक्ष RPC_CMD_GET_TENSOR:        प्रत्यागम "GET_TENSOR";
        पक्ष RPC_CMD_COPY_TENSOR:       प्रत्यागम "COPY_TENSOR";
        पक्ष RPC_CMD_GRAPH_COMPUTE:     प्रत्यागम "GRAPH_COMPUTE";
        पक्ष RPC_CMD_GET_DEVICE_स्मृतिः: प्रत्यागम "GET_DEVICE_MEMORY";
        पक्ष RPC_CMD_INIT_TENSOR:       प्रत्यागम "INIT_TENSOR";
        पक्ष RPC_CMD_GET_ALLOC_आकार:    प्रत्यागम "GET_ALLOC_SIZE";
        पक्ष RPC_CMD_HELLO:             प्रत्यागम "HELLO";
        पक्ष RPC_CMD_DEVICE_गणनम्:      प्रत्यागम "DEVICE_COUNT";
        पक्ष RPC_CMD_GRAPH_RECOMPUTE:   प्रत्यागम "GRAPH_RECOMPUTE";
        पक्ष RPC_CMD_MEMSET_TENSOR:     प्रत्यागम "MEMSET_TENSOR";
        पक्ष RPC_CMD_समुच्चयः_TENSOR_2_D:     प्रत्यागम "SET_TENSOR_2D";
        पक्ष RPC_CMD_GET_TENSOR_2D:     प्रत्यागम "GET_TENSOR_2D";
        पक्ष RPC_CMD_COMM_INIT:         प्रत्यागम "COMM_INIT";
        पक्ष RPC_CMD_COMM_ALLREDUCE:    प्रत्यागम "COMM_ALLREDUCE";
        पक्ष RPC_CMD_COMM_FREE:         प्रत्यागम "COMM_FREE";
        पक्ष RPC_CMD_NONE:              प्रत्यागम "NONE";
        स्वभाव:                        प्रत्यागम "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ध्रुव size_t HASH_देहली = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ध्रुव size_t GRAPH_CACHE_MAX = 1024;

संरचना rpc_सन्देशः_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_आकार];
};

संरचना rpc_सन्देशः_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_आकार];
};

संरचना rpc_सन्देशः_device_गणनम्_rsp {
    uint32_t device_गणनम्;
};

संरचना rpc_सन्देशः_get_alloc_आकार_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

संरचना rpc_सन्देशः_get_alloc_आकार_rsp {
    uint64_t alloc_आकार;
};

संरचना rpc_सन्देशः_init_tensor_req {
    rpc_tensor tensor;
};

संरचना rpc_सन्देशः_alloc_बफर_req {
    uint32_t device;
    uint64_t size;
};

संरचना rpc_सन्देशः_alloc_बफर_rsp {
    uint64_t remote_ptr;
    uint64_t remote_आकार;
};

संरचना rpc_सन्देशः_get_alignment_req {
    uint32_t device;
};

संरचना rpc_सन्देशः_get_alignment_rsp {
    uint64_t alignment;
};

संरचना rpc_सन्देशः_get_max_आकार_req {
    uint32_t device;
};

संरचना rpc_सन्देशः_get_max_आकार_rsp {
    uint64_t max_size;
};

संरचना rpc_सन्देशः_बफर_get_आधारः_req {
    uint64_t remote_ptr;
};

संरचना rpc_सन्देशः_बफर_get_आधारः_rsp {
    uint64_t आधारः_ptr;
};

संरचना rpc_सन्देशः_free_बफर_req {
    uint64_t remote_ptr;
};

संरचना rpc_सन्देशः_बफर_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

संरचना rpc_सन्देशः_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t विस्थापनम्;
    uint64_t size;
    uint8_t value;
};

संरचना rpc_सन्देशः_समुच्चयः_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t विस्थापनम्;
    uint64_t hash;
};

संरचना rpc_सन्देशः_समुच्चयः_tensor_hash_rsp {
    uint8_t परिणाम;
};

संरचना rpc_सन्देशः_get_tensor_req {
    rpc_tensor tensor;
    uint64_t विस्थापनम्;
    uint64_t size;
};

संरचना rpc_सन्देशः_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

संरचना rpc_सन्देशः_copy_tensor_rsp {
    uint8_t परिणाम;
};

संरचना rpc_सन्देशः_get_device_स्मृतिः_req {
    uint32_t device;
};

संरचना rpc_सन्देशः_get_device_स्मृतिः_rsp {
    uint64_t free_mem;
    uint64_t योग_mem;
};

संरचना rpc_सन्देशः_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

संरचना rpc_सन्देशः_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t विस्थापनम्;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

संरचना rpc_सन्देशः_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t पत्तनम्;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    वर्ण     host[64];  // rank > 0: rank 0's host
};

संरचना rpc_सन्देशः_comm_init_rsp {
    uint8_t ok;
};

संरचना rpc_सन्देशः_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

संरचना rpc_सन्देशः_comm_free_req {
    uint32_t device;
};

#निर्देशः pack(निष्कासय)

// RPC data structures

स्थिर ggml_guid_t ggml_backend_rpc_guid() {
    स्थिर ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    प्रत्यागम &guid;
}

संरचना ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

संरचना ggml_backend_rpc_बफर_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

वर्ग rpc_dispatcher;
संरचना ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

संरचना ggml_backend_rpc_बफर_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    शून्य                            * आधारः_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
स्थिर uint64_t fnv_hash(ध्रुव uint8_t * दत्त, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ध्रुव uint64_t fnv_अविभाज्यः = 0x100000001b3ULL;

    प्रति (size_t i = 0; i < len; ++i) {
        hash ^= दत्त[i];
        hash *= fnv_अविभाज्यः;
    }
    प्रत्यागम hash;
}

स्थिर तार्किक send_सन्देशः(socket_ptr sock, ध्रुव शून्य * सन्देशः_2, size_t सन्देशः_आकार) {
    यदि (!sock->send_दत्त(&सन्देशः_आकार, परिमाण(सन्देशः_आकार))) {
        प्रत्यागम असत्य;
    }
    यदि (!sock->send_दत्त(सन्देशः_2, सन्देशः_आकार)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम sock->flush();
}

स्थिर तार्किक recv_सन्देशः(socket_ptr sock, शून्य * सन्देशः_2, size_t सन्देशः_आकार) {
    uint64_t size;
    यदि (!sock->recv_दत्त(&size, परिमाण(size))) {
        प्रत्यागम असत्य;
    }
    यदि (size != सन्देशः_आकार) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम sock->recv_दत्त(सन्देशः_2, सन्देशः_आकार);
}

स्थिर तार्किक recv_सन्देशः(socket_ptr sock, std::vector<uint8_t> & निवेशः) {
    uint64_t size;
    यदि (!sock->recv_दत्त(&size, परिमाण(size))) {
        प्रत्यागम असत्य;
    }
    यत्न {
        निवेशः.resize(size);
    } गृहाण (ध्रुव std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        प्रत्यागम असत्य;
    }
    प्रत्यागम sock->recv_दत्त(निवेशः.दत्त(), size);
}

स्थिर तार्किक parse_endpoint(ध्रुव std::string & endpoint, std::string & host, पूर्णाङ्क & पत्तनम्) {
    size_t pos = endpoint.find(':');
    यदि (pos == std::string::npos) {
        प्रत्यागम असत्य;
    }
    host = endpoint.substr(0, pos);
    यत्न {
        पत्तनम् = std::stoi(endpoint.substr(pos + 1));
    } गृहाण (...) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
स्थिर तार्किक send_rpc_cmd(socket_ptr sock, गणना rpc_cmd cmd, ध्रुव शून्य * निवेशः, size_t निवेशः_आकार) {
    uint8_t cmd_byte = cmd;
    यदि (!sock->send_दत्त(&cmd_byte, परिमाण(cmd_byte))) {
        प्रत्यागम असत्य;
    }
    यदि (!sock->send_दत्त(&निवेशः_आकार, परिमाण(निवेशः_आकार))) {
        प्रत्यागम असत्य;
    }
    यदि (!sock->send_दत्त(निवेशः, निवेशः_आकार)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
स्थिर तार्किक send_rpc_cmd(socket_ptr sock, गणना rpc_cmd cmd, ध्रुव शून्य * निवेशः, size_t निवेशः_आकार, शून्य * output, size_t output_आकार) {
    यदि (!send_rpc_cmd(sock, cmd, निवेशः, निवेशः_आकार)) {
        प्रत्यागम असत्य;
    }
    uint64_t बहिर्गमनम्_आकार;
    यदि (!sock->recv_दत्त(&बहिर्गमनम्_आकार, परिमाण(बहिर्गमनम्_आकार))) {
        प्रत्यागम असत्य;
    }
    यदि (बहिर्गमनम्_आकार != output_आकार) {
        प्रत्यागम असत्य;
    }
    यदि (!sock->recv_दत्त(output, output_आकार)) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
स्थिर constexpr स्वचालित RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

स्थिर अन्तःस्थ शून्य rpc_cpu_relax() {
#यदि defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ चञ्चल("yield" ::: "memory");
#अन्यथा
    std::this_thread::yield();
#यदि_अन्तः
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
स्थिर तार्किक negotiate_hello(ध्रुव std::shared_ptr<socket_t> & sock) {
    rpc_सन्देशः_hello_req request = {};
    rpc_सन्देशः_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    तार्किक स्थितिः = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, परिमाण(request), &response, परिमाण(response));
    RPC_स्थितिः_ASSERT(स्थितिः);

    यदि (response.major != RPC_PROTO_MAJOR_संस्करणम् || response.minor > RPC_PROTO_MINOR_संस्करणम्) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        प्रत्यागम असत्य;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    प्रत्यागम सत्य;
}

प्रतिरूपम् <प्रकारनाम T>
वर्ग सन्देशः_queue {
सार्वजनिक:
    सन्देशः_queue() {}

    तार्किक प्रक्षिप(ध्रुव T &value) {
        std::unique_lock<std::mutex> तालकम्(mutex);
        यदि (interrupted) {
            प्रत्यागम असत्य;
        }
        queue.प्रक्षिप(value);
        cvar.notify_all();
        प्रत्यागम सत्य;
    }

    तार्किक निष्कासय(T* बहिर्गमनम्) {
        std::unique_lock<std::mutex> तालकम्(mutex);
        cvar.प्रतीक्षस्व(तालकम्, [एतत्] { प्रत्यागम !queue.रिक्तानि() || interrupted; });
        यदि (interrupted) {
            प्रत्यागम असत्य;
        }
        *बहिर्गमनम् = queue.front();
        queue.निष्कासय();
        प्रत्यागम सत्य;
    }

    तार्किक try_निष्कासय(T* बहिर्गमनम्) {
        std::unique_lock<std::mutex> तालकम्(mutex);
        यदि (interrupted || queue.रिक्तानि()) {
            प्रत्यागम असत्य;
        }
        *बहिर्गमनम् = queue.front();
        queue.निष्कासय();
        प्रत्यागम सत्य;
    }

    शून्य व्यवधानम्() {
        std::unique_lock<std::mutex> तालकम्(mutex);
        interrupted = सत्य;
        तालकम्.unlock();
        cvar.notify_all();
    }

गुप्त:
    तार्किक interrupted = असत्य;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

वर्ग rpc_dispatcher {
सार्वजनिक:
    rpc_dispatcher() {
    }

    शून्य send(गणना rpc_cmd cmd, std::shared_ptr<ध्रुव शून्य> निवेशः, size_t निवेशः_आकार);
    शून्य send(गणना rpc_cmd cmd, std::shared_ptr<ध्रुव शून्य> निवेशः, size_t निवेशः_आकार, शून्य * output, size_t output_आकार);
    शून्य send_async(गणना rpc_cmd cmd, std::shared_ptr<ध्रुव शून्य> निवेशः, size_t निवेशः_आकार);
    शून्य send_async(गणना rpc_cmd cmd, std::shared_ptr<ध्रुव शून्य> निवेशः, size_t निवेशः_आकार, शून्य * output, size_t output_आकार);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    शून्य event_free(ggml_backend_event_t event);
    शून्य event_synchronize(ggml_backend_event_t event);
    शून्य event_लेखः(ggml_backend_event_t event);
    शून्य synchronize();
    शून्य busy_spin_acquire();
    शून्य busy_spin_release();
    शून्य graph_compute(uint32_t device, ध्रुव ggml_cgraph * cgraph);

    शून्य start(ध्रुव std::string & endpoint);
    शून्य work();

    ~rpc_dispatcher();

गुप्त:
    संरचना rpc_सन्देशः {
        rpc_cmd                       cmd;
        std::shared_ptr<ध्रुव शून्य>   निवेशः;
        size_t                        निवेशः_आकार;
        शून्य                        * output;
        size_t                        output_आकार;
        std::promise<शून्य>            completion;
    };
    उपयुज्य rpc_सन्देशः_ptr   = std::shared_ptr<rpc_सन्देशः>;
    उपयुज्य rpc_सन्देशः_queue = सन्देशः_queue<rpc_सन्देशः_ptr>;
    संरचना rpc_event {
        rpc_सन्देशः_ptr              सन्देशः_2;
        std::shared_future<शून्य> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_सन्देशः_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

स्थिर शून्य rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

शून्य rpc_dispatcher::send(गणना rpc_cmd cmd, std::shared_ptr<ध्रुव शून्य> निवेशः, size_t निवेशः_आकार) {
    स्वचालित सन्देशः_2 = std::make_shared<rpc_सन्देशः>();
    सन्देशः_2->cmd = cmd;
    सन्देशः_2->निवेशः = निवेशः;
    सन्देशः_2->निवेशः_आकार = निवेशः_आकार;
    सन्देशः_2->output = शून्यसूचकः;
    सन्देशः_2->output_आकार = 0;
    GGML_ASSERT(queue.प्रक्षिप(सन्देशः_2));
    स्वचालित future = सन्देशः_2->completion.get_future();
    future.प्रतीक्षस्व();
}

शून्य rpc_dispatcher::send_async(गणना rpc_cmd cmd, std::shared_ptr<ध्रुव शून्य> निवेशः, size_t निवेशः_आकार) {
    स्वचालित सन्देशः_2 = std::make_shared<rpc_सन्देशः>();
    सन्देशः_2->cmd = cmd;
    सन्देशः_2->निवेशः = निवेशः;
    सन्देशः_2->निवेशः_आकार = निवेशः_आकार;
    सन्देशः_2->output = शून्यसूचकः;
    सन्देशः_2->output_आकार = 0;
    GGML_ASSERT(queue.प्रक्षिप(सन्देशः_2));
}

शून्य rpc_dispatcher::send(गणना rpc_cmd cmd, std::shared_ptr<ध्रुव शून्य> निवेशः, size_t निवेशः_आकार, शून्य * output, size_t output_आकार) {
    स्वचालित सन्देशः_2 = std::make_shared<rpc_सन्देशः>();
    सन्देशः_2->cmd = cmd;
    सन्देशः_2->निवेशः = निवेशः;
    सन्देशः_2->निवेशः_आकार = निवेशः_आकार;
    सन्देशः_2->output = output;
    सन्देशः_2->output_आकार = output_आकार;
    GGML_ASSERT(queue.प्रक्षिप(सन्देशः_2));
    स्वचालित future = सन्देशः_2->completion.get_future();
    future.प्रतीक्षस्व();
}

शून्य rpc_dispatcher::send_async(गणना rpc_cmd cmd, std::shared_ptr<ध्रुव शून्य> निवेशः, size_t निवेशः_आकार, शून्य * output, size_t output_आकार) {
    स्वचालित सन्देशः_2 = std::make_shared<rpc_सन्देशः>();
    सन्देशः_2->cmd = cmd;
    सन्देशः_2->निवेशः = निवेशः;
    सन्देशः_2->निवेशः_आकार = निवेशः_आकार;
    सन्देशः_2->output = output;
    सन्देशः_2->output_आकार = output_आकार;
    GGML_ASSERT(queue.प्रक्षिप(सन्देशः_2));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = नूतन rpc_event;
    ev->सन्देशः_2 = std::make_shared<rpc_सन्देशः>();
    ev->सन्देशः_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->सन्देशः_2->completion.get_future().share();
    GGML_ASSERT(queue.प्रक्षिप(ev->सन्देशः_2));
    प्रत्यागम नूतन ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

शून्य rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    विलोपय ev;
}

शून्य rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.प्रतीक्षस्व();
}

शून्य rpc_dispatcher::event_लेखः(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->सन्देशः_2 = std::make_shared<rpc_सन्देशः>();
    ev->सन्देशः_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->सन्देशः_2->completion.get_future().share();
    GGML_ASSERT(queue.प्रक्षिप(ev->सन्देशः_2));
}

शून्य rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    स्वचालित सन्देशः_2 = std::make_shared<rpc_सन्देशः>();
    सन्देशः_2->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.प्रक्षिप(सन्देशः_2));
    सन्देशः_2->completion.get_future().प्रतीक्षस्व();
}

शून्य rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

शून्य rpc_dispatcher::busy_spin_release() {
    ध्रुव अचिह्नित previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

शून्य rpc_dispatcher::start(ध्रुव std::string & endpoint) {
    std::string host;
    पूर्णाङ्क पत्तनम्;
    यदि (!parse_endpoint(endpoint, host, पत्तनम्)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    यदि (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), पत्तनम्);
    यदि (sock == शून्यसूचकः) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    यदि (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = सत्य;
    thread = std::thread(rpc_dispatcher_trampoline, एतत्);
}

शून्य rpc_dispatcher::work() {
    स्वचालित चरमः_cmd = std::chrono::steady_clock::now();
    यावत् (running) {
        rpc_सन्देशः_ptr सन्देशः_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ध्रुव तार्किक spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - चरमः_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        यदि (spin) {
            यदि (!queue.try_निष्कासय(&सन्देशः_ptr)) {
                rpc_cpu_relax();
                अनुवर्त;
            }
        } अन्यथा यदि (!queue.निष्कासय(&सन्देशः_ptr)) {
            भङ्ग;
        }
        यदि (सन्देशः_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(सन्देशः_ptr->cmd),
                     rpc_log_format_आकार(सन्देशः_ptr->निवेशः_आकार).c_str(), rpc_log_format_आकार(सन्देशः_ptr->output_आकार).c_str());
            ध्रुव तार्किक trace = rpc_log_debug_level() >= 3;
            ध्रुव स्वचालित t0 = std::chrono::steady_clock::now();
            यदि (सन्देशः_ptr->output) {
                तार्किक स्थितिः = send_rpc_cmd(sock, सन्देशः_ptr->cmd, सन्देशः_ptr->निवेशः.get(), सन्देशः_ptr->निवेशः_आकार, सन्देशः_ptr->output, सन्देशः_ptr->output_आकार);
                RPC_स्थितिः_ASSERT(स्थितिः);
            } अन्यथा {
                तार्किक स्थितिः = send_rpc_cmd(sock, सन्देशः_ptr->cmd, सन्देशः_ptr->निवेशः.get(), सन्देशः_ptr->निवेशः_आकार);
                RPC_स्थितिः_ASSERT(स्थितिः);
            }
            यदि (trace) {
                ध्रुव std::chrono::duration<द्विगुण, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(सन्देशः_ptr->cmd), elapsed.count());
            }
        } अन्यथा {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        सन्देशः_ptr->completion.set_value();
        चरमः_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = असत्य;
    queue.व्यवधानम्();
    sock = शून्यसूचकः;
    यदि (thread.joinable()) {
        thread.join();
    }
}

स्थिर std::shared_ptr<rpc_dispatcher> get_dispatcher(ध्रुव std::string & endpoint) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> तालकम्(mutex);
    स्थिर std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    स्वचालित it = dispatchers.find(endpoint);
    यदि (it != dispatchers.end()) {
        यदि (स्वचालित dispatcher = it->second.तालकम्()) {
            प्रत्यागम dispatcher;
        }
    }

    स्वचालित dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    प्रत्यागम dispatcher;
}

स्थिर शून्य ggml_backend_rpc_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    स्वचालित request = std::make_shared<rpc_सन्देशः_free_बफर_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_बफर, request, परिमाण(*request));
    विलोपय ctx;
}

स्थिर शून्य * ggml_backend_rpc_बफर_get_आधारः(ggml_backend_बफर_t बफर) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    यदि (ctx->आधारः_ptr != शून्यसूचकः) {
        प्रत्यागम ctx->आधारः_ptr;
    }
    स्वचालित request = std::make_shared<rpc_सन्देशः_बफर_get_आधारः_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_सन्देशः_बफर_get_आधारः_rsp response;
    ctx->dispatcher->send(RPC_CMD_बफर_GET_आधारः, request, परिमाण(*request), &response, परिमाण(response));
    ctx->आधारः_ptr = reinterpret_cast<शून्य *>(response.आधारः_ptr);
    प्रत्यागम ctx->आधारः_ptr;
}

स्थिर तार्किक ggml_backend_बफर_is_rpc(ggml_backend_बफर_t बफर) {
    प्रत्यागम बफर->iface.free_बफर == ggml_backend_rpc_बफर_free_बफर;
}

स्थिर rpc_tensor serialize_tensor(ध्रुव ggml_tensor * tensor, ध्रुव rpc_dispatcher * dispatcher = शून्यसूचकः) {
    rpc_tensor परिणाम;
    यदि (!tensor) {
        स्मृतिं_पूरय(&परिणाम, 0, परिमाण(परिणाम));
        प्रत्यागम परिणाम;
    }

    परिणाम.id = reinterpret_cast<uint64_t>(tensor);
    परिणाम.type = tensor->type;
    यदि (tensor->बफर && ggml_backend_बफर_is_rpc(tensor->बफर)) {
        ggml_backend_बफर_t बफर = tensor->बफर;
        ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        यदि (ctx != शून्यसूचकः && (dispatcher == शून्यसूचकः || ctx->dispatcher.get() == dispatcher)) {
            परिणाम.बफर = ctx->remote_ptr;
            परिणाम.दत्त = reinterpret_cast<uint64_t>(tensor->दत्त);
        } अन्यथा {
            परिणाम.बफर = 0;
            परिणाम.दत्त = 0;
        }
    } अन्यथा {
        परिणाम.बफर = 0;
        परिणाम.दत्त   = 0;
    }
    प्रति (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम.ne[i] = tensor->ne[i];
        परिणाम.nb[i] = tensor->nb[i];
    }
    परिणाम.op = tensor->op;
    प्रति (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / परिमाण(int32_t); i++) {
        परिणाम.op_params[i] = tensor->op_params[i];
    }
    परिणाम.flags = tensor->flags;
    प्रति (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        परिणाम.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    परिणाम.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    परिणाम.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    स्मृतिं_पूरय(परिणाम.name, 0, परिमाण(परिणाम.name));
    परिणाम.use_गणनम् = 0;

    snprintf(परिणाम.name, GGML_MAX_NAME, "%s", tensor->name);
    प्रत्यागम परिणाम;
}

स्थिर गणना ggml_स्थितिः ggml_backend_rpc_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    यदि (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == शून्यसूचकः)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        स्वचालित request = std::make_shared<rpc_सन्देशः_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, परिमाण(*request));
    }
    प्रत्यागम GGML_स्थितिः_SUCCESS;
}

स्थिर शून्य ggml_backend_rpc_बफर_memset_tensor(
        ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value, size_t विस्थापनम्, size_t size) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    स्वचालित request = std::make_shared<rpc_सन्देशः_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->विस्थापनम् = विस्थापनम्;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), विस्थापनम्,
             rpc_log_format_आकार(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, परिमाण(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
स्थिर std::shared_ptr<uint8_t> serialize_समुच्चयः_tensor(ध्रुव rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t विस्थापनम्, ध्रुव शून्य * दत्त, size_t size, size_t & निवेशः_आकार) {
    निवेशः_आकार = परिमाण(rpc_tensor) + परिमाण(cache_flag) + परिमाण(विस्थापनम्) + size;
    uint8_t * निवेशः = नूतन uint8_t[निवेशः_आकार]();
    uint8_t * p = निवेशः;
    स्मृतिं_प्रतिलिख(p, &rpc_tensor, परिमाण(rpc_tensor)); p += परिमाण(rpc_tensor);
    स्मृतिं_प्रतिलिख(p, &cache_flag, परिमाण(cache_flag)); p += परिमाण(cache_flag);
    स्मृतिं_प्रतिलिख(p, &विस्थापनम्,     परिमाण(विस्थापनम्));     p += परिमाण(विस्थापनम्);
    स्मृतिं_प्रतिलिख(p, दत्त, size);
    प्रत्यागम std::shared_ptr<uint8_t>(निवेशः, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
स्थिर तार्किक rpc_use_hash_cache(ध्रुव ggml_tensor * tensor, size_t size) {
    प्रत्यागम size > HASH_देहली && tensor->बफर->usage == GGML_BACKEND_बफर_USAGE_WEIGHTS;
}

स्थिर शून्य ggml_backend_rpc_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), विस्थापनम्,
             rpc_log_format_आकार(size).c_str());
    यदि (rpc_use_hash_cache(tensor, size)) {
        स्वचालित request = std::make_shared<rpc_सन्देशः_समुच्चयः_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->विस्थापनम् = विस्थापनम्;
        request->hash = fnv_hash((ध्रुव uint8_t*)दत्त, size);
        rpc_सन्देशः_समुच्चयः_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_समुच्चयः_TENSOR_HASH, request, परिमाण(*request), &response, परिमाण(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.परिणाम ? "hit" : "miss");
        यदि (response.परिणाम) {
            // the server has the same data, no need to send it
            प्रत्यागम;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t निवेशः_आकार;
    स्वचालित निवेशः = serialize_समुच्चयः_tensor(rpc_tensor, cache_flag, विस्थापनम्, दत्त, size, निवेशः_आकार);
    ctx->dispatcher->send(RPC_CMD_समुच्चयः_TENSOR, निवेशः, निवेशः_आकार);
}

स्थिर शून्य ggml_backend_rpc_बफर_समुच्चयः_tensor_2d(ggml_backend_बफर_t बफर, ggml_tensor * tensor, ध्रुव शून्य * दत्त,
        size_t विस्थापनम्, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_दत्त) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), विस्थापनम्, rpc_log_format_आकार(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t निवेशः_आकार = परिमाण(rpc_tensor) + 4*परिमाण(uint64_t) + size*n_copies;
    uint8_t * निवेशः = नूतन uint8_t[निवेशः_आकार]();
    uint8_t * dest = निवेशः;
    स्मृतिं_प्रतिलिख(dest, &rpc_tensor, परिमाण(rpc_tensor));
    dest += परिमाण(rpc_tensor);
    uint64_t शीर्षकम्[4] = { विस्थापनम्, size, n_copies, stride_tensor };
    स्मृतिं_प्रतिलिख(dest, शीर्षकम्, परिमाण(शीर्षकम्));
    dest += परिमाण(शीर्षकम्);
    प्रति (size_t i = 0; i < n_copies; i++) {
        स्मृतिं_प्रतिलिख(dest + i*size, (ध्रुव वर्ण *)दत्त + i*stride_दत्त, size);
    }
    std::shared_ptr<uint8_t> निवेशः_ptr(निवेशः, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_समुच्चयः_TENSOR_2_D, निवेशः_ptr, निवेशः_आकार);
}

स्थिर शून्य ggml_backend_rpc_बफर_get_tensor_2d(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * tensor, शून्य * दत्त,
        size_t विस्थापनम्, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_दत्त) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), विस्थापनम्, rpc_log_format_आकार(size).c_str(), n_copies, stride_tensor);
    स्वचालित request = std::make_shared<rpc_सन्देशः_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->विस्थापनम्   = विस्थापनम्;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    यदि (stride_दत्त == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, परिमाण(*request), दत्त, size*n_copies);
    } अन्यथा {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, परिमाण(*request), packed.दत्त(), packed.size());
        प्रति (size_t i = 0; i < n_copies; i++) {
            स्मृतिं_प्रतिलिख((वर्ण *)दत्त + i*stride_दत्त, packed.दत्त() + i*size, size);
        }
    }
}

स्थिर शून्य ggml_backend_rpc_बफर_get_tensor(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), विस्थापनम्,
             rpc_log_format_आकार(size).c_str());
    स्वचालित request = std::make_shared<rpc_सन्देशः_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->विस्थापनम् = विस्थापनम्;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, परिमाण(*request), दत्त, size);
}

स्थिर तार्किक ggml_backend_rpc_बफर_cpy_tensor(ggml_backend_बफर_t बफर, ध्रुव ggml_tensor * src, ggml_tensor * dst) {
    यदि (ggml_backend_बफर_is_rpc(src->बफर)) {
        // check if src and dst are on the same server
        ggml_backend_बफर_t src_बफर = src->बफर;
        ggml_backend_rpc_बफर_context * src_ctx = (ggml_backend_rpc_बफर_context *)src_बफर->context;
        ggml_backend_बफर_t dst_बफर = dst->बफर;
        ggml_backend_rpc_बफर_context * dst_ctx = (ggml_backend_rpc_बफर_context *)dst_बफर->context;
        यदि (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            प्रत्यागम असत्य;
        }
        ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        स्वचालित request = std::make_shared<rpc_सन्देशः_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_सन्देशः_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, परिमाण(*request), &response, परिमाण(response));
        प्रत्यागम response.परिणाम;
    }
    प्रत्यागम असत्य;
}

स्थिर शून्य ggml_backend_rpc_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_rpc_बफर_context * ctx = (ggml_backend_rpc_बफर_context *)बफर->context;
    स्वचालित request = std::make_shared<rpc_सन्देशः_बफर_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_बफर_CLEAR, request, परिमाण(*request));
}

स्थिर ggml_backend_बफर_i ggml_backend_rpc_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_rpc_बफर_get_आधारः,
    /* .init_tensor     = */ ggml_backend_rpc_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_बफर_समुच्चयः_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_बफर_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_बफर_समुच्चयः_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_बफर_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_बफर_clear,
    /* .reset           = */ रिक्त,
};

स्थिर ध्रुव वर्ण * ggml_backend_rpc_बफर_type_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    प्रत्यागम buft_ctx->name.c_str();
}

स्थिर ggml_backend_बफर_t ggml_backend_rpc_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t size) {
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    स्वचालित request = std::make_shared<rpc_सन्देशः_alloc_बफर_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_सन्देशः_alloc_बफर_rsp response;

    स्वचालित dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_बफर, request, परिमाण(*request), &response, परिमाण(response));
    यदि (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_आकार(request->size).c_str(),
                response.remote_ptr, rpc_log_format_आकार(response.remote_आकार).c_str());
        ggml_backend_बफर_t बफर = ggml_backend_बफर_init(buft,
            ggml_backend_rpc_बफर_interface,
            नूतन ggml_backend_rpc_बफर_context{dispatcher, शून्यसूचकः, response.remote_ptr},
            response.remote_आकार);
        प्रत्यागम बफर;
    } अन्यथा {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_आकार(request->size).c_str());
        प्रत्यागम शून्यसूचकः;
    }
}

स्थिर size_t get_alignment(ध्रुव std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    स्वचालित request = std::make_shared<rpc_सन्देशः_get_alignment_req>();
    request->device = device;
    rpc_सन्देशः_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, परिमाण(*request), &response, परिमाण(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_आकार(response.alignment).c_str());
    प्रत्यागम response.alignment;
}

स्थिर size_t ggml_backend_rpc_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    प्रत्यागम buft_ctx->alignment;
}

स्थिर size_t get_max_आकार(ध्रुव std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    स्वचालित request = std::make_shared<rpc_सन्देशः_get_max_आकार_req>();
    request->device = device;
    rpc_सन्देशः_get_max_आकार_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_आकार, request, परिमाण(*request), &response, परिमाण(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_आकार(response.max_size).c_str());
    प्रत्यागम response.max_size;
}

स्थिर size_t ggml_backend_rpc_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    प्रत्यागम buft_ctx->max_size;
}

स्थिर size_t ggml_backend_rpc_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, ध्रुव ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    तार्किक rpc_get = असत्य;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == शून्यसूचकः);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_आकार_may_expand(tensor->op);

    यदि (rpc_get) {
        ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ध्रुव size_t min_आकार = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        संरचना alloc_आकार_cache_कुञ्चिका {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / परिमाण(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_आकार_cache_कुञ्चिका कुञ्चिका = {};
        कुञ्चिका.device = buft_ctx->device;
        कुञ्चिका.type = tensor->type;
        कुञ्चिका.op = tensor->op;
        स्मृतिं_प्रतिलिख(कुञ्चिका.op_params, tensor->op_params, परिमाण(कुञ्चिका.op_params));
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
            कुञ्चिका.ne[i] = (uint32_t)tensor->ne[i];
            कुञ्चिका.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ध्रुव uint8_t *)&कुञ्चिका, परिमाण(कुञ्चिका));
        cache_hash = fnv_hash((ध्रुव uint8_t *)buft_ctx->endpoint.दत्त(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        स्थिर std::mutex cache_mutex;
        स्थिर std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> तालकम्(cache_mutex);
            स्वचालित it = cache.find(cache_hash);
            यदि (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_आकार(it->second).c_str());
                प्रत्यागम std::max<size_t>(it->second, min_आकार);
            }
        }

        स्वचालित request = std::make_shared<rpc_सन्देशः_get_alloc_आकार_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_सन्देशः_get_alloc_आकार_rsp response;
        स्वचालित dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_आकार, request, परिमाण(*request), &response, परिमाण(response));

        {
            std::lock_guard<std::mutex> तालकम्(cache_mutex);
            cache[cache_hash] = response.alloc_आकार;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_आकार(response.alloc_आकार).c_str());

        प्रत्यागम std::max<size_t>(response.alloc_आकार, min_आकार);
    }

    प्रत्यागम ggml_nbytes(tensor);
}

स्थिर ggml_backend_बफर_type_i ggml_backend_rpc_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_बफर_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_rpc_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_rpc_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};

स्थिर ध्रुव वर्ण * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    प्रत्यागम rpc_ctx->name.c_str();
}

स्थिर शून्य ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    विलोपय rpc_ctx;
    विलोपय backend;
}

स्थिर शून्य ggml_backend_rpc_समुच्चयः_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ध्रुव शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), विस्थापनम्,
             rpc_log_format_आकार(size).c_str());
    यदि (rpc_use_hash_cache(tensor, size)) {
        स्वचालित request = std::make_shared<rpc_सन्देशः_समुच्चयः_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->विस्थापनम् = विस्थापनम्;
        request->hash = fnv_hash((ध्रुव uint8_t*)दत्त, size);
        rpc_सन्देशः_समुच्चयः_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_समुच्चयः_TENSOR_HASH, request, परिमाण(*request), &response, परिमाण(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.परिणाम ? "hit" : "miss");
        यदि (response.परिणाम) {
            // the server has the same data, no need to send it
            प्रत्यागम;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t निवेशः_आकार;
    स्वचालित निवेशः = serialize_समुच्चयः_tensor(rpc_tensor, cache_flag, विस्थापनम्, दत्त, size, निवेशः_आकार);
    ctx->dispatcher->send_async(RPC_CMD_समुच्चयः_TENSOR, निवेशः, निवेशः_आकार);
}

स्थिर शून्य ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ध्रुव ggml_tensor * tensor, शून्य * दत्त, size_t विस्थापनम्, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), विस्थापनम्,
             rpc_log_format_आकार(size).c_str());
    स्वचालित request = std::make_shared<rpc_सन्देशः_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->विस्थापनम् = विस्थापनम्;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, परिमाण(*request), दत्त, size);
}

स्थिर शून्य ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

स्थिर शून्य सङ्कलय_tensor(ggml_tensor * tensor, ध्रुव ggml_cgraph * cgraph, ध्रुव rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    यदि (tensor == शून्यसूचकः) {
        प्रत्यागम;
    }
    यदि (visited.find(tensor) != visited.end()) {
        प्रत्यागम;
    }
    visited.insert(tensor);
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        सङ्कलय_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    सङ्कलय_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor परिणाम = serialize_tensor(tensor, dispatcher);
    ध्रुव size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_समुच्चयः, tensor);
    यदि (hash_pos != GGML_HASHSET_पूर्णम् && ggml_bitset_get(cgraph->visited_hash_समुच्चयः.प्रयुक्तम्, hash_pos)) {
        परिणाम.use_गणनम् = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(परिणाम);
}

स्थिर uint8_t * serialize_graph(uint32_t device, ध्रुव ggml_cgraph * cgraph, ध्रुव rpc_dispatcher * dispatcher, size_t * output_आकार) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    प्रति (uint32_t i = 0; i < n_nodes; i++) {
        सङ्कलय_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_आकार = 2*परिमाण(uint32_t) + परिमाण(uint64_t) + n_nodes * परिमाण(uint64_t) + परिमाण(uint32_t) + n_tensors * परिमाण(rpc_tensor);
    uint8_t * output = नूतन uint8_t[*output_आकार]();
    uint8_t * dest = output;
    स्मृतिं_प्रतिलिख(dest, &device, परिमाण(device));
    dest += परिमाण(device);
    स्मृतिं_प्रतिलिख(dest, &cgraph->uid, परिमाण(cgraph->uid));
    dest += परिमाण(cgraph->uid);
    स्मृतिं_प्रतिलिख(dest, &n_nodes, परिमाण(n_nodes));
    dest += परिमाण(n_nodes);
    प्रति (uint32_t i = 0; i < n_nodes; i++) {
        स्मृतिं_प्रतिलिख(dest + i * परिमाण(uint64_t), &cgraph->nodes[i], परिमाण(uint64_t));
    }
    dest += n_nodes * परिमाण(uint64_t);
    स्मृतिं_प्रतिलिख(dest, &n_tensors, परिमाण(n_tensors));
    dest += परिमाण(n_tensors);
    rpc_tensor * बहिर्गमनम्_tensors = (rpc_tensor *)dest;
    स्मृतिं_प्रतिलिख(बहिर्गमनम्_tensors, tensors.दत्त(), n_tensors * परिमाण(rpc_tensor));
    प्रत्यागम output;
}

शून्य rpc_dispatcher::graph_compute(uint32_t device, ध्रुव ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> तालकम्(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    स्वचालित & device_graph_uids = graph_uids[device];
    तार्किक reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    यदि (reuse) {
        स्वचालित request = std::make_shared<rpc_सन्देशः_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, परिमाण(*request));
    } अन्यथा {
        यदि (cgraph->uid != 0) {
            यदि (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t निवेशः_आकार = 0;
        uint8_t * निवेशः = serialize_graph(device, cgraph, एतत्, &निवेशः_आकार);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_आकार(निवेशः_आकार).c_str());
        std::shared_ptr<uint8_t> निवेशः_ptr(निवेशः, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, निवेशः_ptr, निवेशः_आकार);
    }
}

स्थिर गणना ggml_स्थितिः ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    प्रत्यागम GGML_स्थितिः_SUCCESS;
}

स्थिर शून्य ggml_backend_rpc_event_लेखः(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_लेखः(event);
}

स्थिर शून्य ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_अप्रयुक्तम्(backend);
    GGML_अप्रयुक्तम्(event);
}

स्थिर शून्य ggml_backend_rpc_समुच्चयः_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ध्रुव शून्य * दत्त,
        size_t विस्थापनम्, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_दत्त) {
    ggml_backend_tensor_समुच्चयः_2d(tensor, दत्त, विस्थापनम्, size, n_copies, stride_tensor, stride_दत्त);
    GGML_अप्रयुक्तम्(backend);
}

स्थिर शून्य ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ध्रुव ggml_tensor * tensor, शून्य * दत्त,
        size_t विस्थापनम्, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_दत्त) {
    ggml_backend_tensor_get_2d(tensor, दत्त, विस्थापनम्, size, n_copies, stride_tensor, stride_दत्त);
    GGML_अप्रयुक्तम्(backend);
}

स्थिर ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_समुच्चयः_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_समुच्चयः_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ रिक्त,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_लेखः,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ रिक्त,
};

ggml_backend_बफर_type_t ggml_backend_rpc_बफर_type(ध्रुव वर्ण * endpoint, uint32_t device) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> तालकम्(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    स्थिर std::unordered_map<std::string, ggml_backend_बफर_type_t> buft_map;
    स्वचालित it = buft_map.find(buft_name);
    यदि (it != buft_map.end()) {
        प्रत्यागम it->second;
    }
    स्वचालित dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_आकार(dispatcher, device);
    ggml_backend_rpc_बफर_type_context * buft_ctx = नूतन ggml_backend_rpc_बफर_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    स्वचालित reg = ggml_backend_rpc_सङ्कलय_server(endpoint);
    ggml_backend_बफर_type_t buft = नूतन ggml_backend_बफर_type {
        /* .iface   = */ ggml_backend_rpc_बफर_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_आकार(alignment).c_str(), rpc_log_format_आकार(max_size).c_str());
    प्रत्यागम buft;
}

ggml_backend_t ggml_backend_rpc_init(ध्रुव वर्ण * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    स्वचालित dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = नूतन ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    स्वचालित reg = ggml_backend_rpc_सङ्कलय_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = नूतन ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    प्रत्यागम backend;
}

तार्किक ggml_backend_is_rpc(ggml_backend_t backend) {
    प्रत्यागम backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

शून्य ggml_backend_rpc_get_device_स्मृतिः(ध्रुव वर्ण * endpoint, uint32_t device, size_t * मुक्त, size_t * योग) {
    स्वचालित dispatcher = get_dispatcher(endpoint);
    स्वचालित request = std::make_shared<rpc_सन्देशः_get_device_स्मृतिः_req>();
    request->device = device;
    rpc_सन्देशः_get_device_स्मृतिः_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_स्मृतिः, request, परिमाण(*request), &response, परिमाण(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_आकार(response.free_mem).c_str(), rpc_log_format_आकार(response.योग_mem).c_str());
    *मुक्त = response.free_mem;
    *योग = response.योग_mem;
}

// RPC server-side implementation

वर्ग rpc_server {
सार्वजनिक:
    rpc_server(std::vector<ggml_backend_t> all_backends, ध्रुव वर्ण * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_अवस्थाः.resize(backends.size());
    }
    ~rpc_server();

    शून्य hello(rpc_सन्देशः_hello_rsp & response);
    तार्किक alloc_बफर(ध्रुव rpc_सन्देशः_alloc_बफर_req & request, rpc_सन्देशः_alloc_बफर_rsp & response);
    तार्किक get_alignment(ध्रुव rpc_सन्देशः_get_alignment_req & request, rpc_सन्देशः_get_alignment_rsp & response);
    तार्किक get_max_आकार(ध्रुव rpc_सन्देशः_get_max_आकार_req & request, rpc_सन्देशः_get_max_आकार_rsp & response);
    तार्किक बफर_get_आधारः(ध्रुव rpc_सन्देशः_बफर_get_आधारः_req & request, rpc_सन्देशः_बफर_get_आधारः_rsp & response);
    तार्किक free_बफर(ध्रुव rpc_सन्देशः_free_बफर_req & request);
    तार्किक बफर_clear(ध्रुव rpc_सन्देशः_बफर_clear_req & request);
    तार्किक memset_tensor(ध्रुव rpc_सन्देशः_memset_tensor_req & request);
    तार्किक समुच्चयः_tensor(ध्रुव std::vector<uint8_t> & निवेशः);
    तार्किक समुच्चयः_tensor_2d(ध्रुव std::vector<uint8_t> & निवेशः);
    तार्किक समुच्चयः_tensor_hash(ध्रुव rpc_सन्देशः_समुच्चयः_tensor_hash_req & request, rpc_सन्देशः_समुच्चयः_tensor_hash_rsp & response);
    तार्किक get_tensor(ध्रुव rpc_सन्देशः_get_tensor_req & request, std::vector<uint8_t> & response);
    तार्किक get_tensor_2d(ध्रुव rpc_सन्देशः_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    तार्किक copy_tensor(ध्रुव rpc_सन्देशः_copy_tensor_req & request, rpc_सन्देशः_copy_tensor_rsp & response);
    तार्किक graph_compute(ध्रुव std::vector<uint8_t> & निवेशः);
    तार्किक graph_recompute(ध्रुव rpc_सन्देशः_graph_recompute_req & request);
    तार्किक comm_init(ध्रुव rpc_सन्देशः_comm_init_req & request, rpc_सन्देशः_comm_init_rsp & response);
    तार्किक comm_allreduce(ध्रुव rpc_सन्देशः_comm_allreduce_req & request);
    तार्किक comm_free(ध्रुव rpc_सन्देशः_comm_free_req & request);
    तार्किक init_tensor(ध्रुव rpc_सन्देशः_init_tensor_req & request);
    तार्किक get_alloc_आकार(ध्रुव rpc_सन्देशः_get_alloc_आकार_req & request, rpc_सन्देशः_get_alloc_आकार_rsp & response);
    तार्किक get_device_स्मृतिः(ध्रुव rpc_सन्देशः_get_device_स्मृतिः_req & request, rpc_सन्देशः_get_device_स्मृतिः_rsp & response);

    संरचना stored_graph {
        std::vector<uint8_t>   बफर;
        ggml_cgraph          * graph;
    };

गुप्त:
    शून्य sync_all_backends();
    तार्किक get_cached_file(uint64_t hash, std::vector<uint8_t> & दत्त);
    ggml_tensor * deserialize_tensor(संरचना ggml_context * ctx, ध्रुव rpc_tensor * tensor);
    ggml_tensor * create_ग्रन्थिः(uint64_t id,
                              संरचना ggml_context * ctx,
                              ध्रुव std::unordered_map<uint64_t, ध्रुव rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, संरचना ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    संरचना comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_बफर_ptr scratch;
        size_t                  scratch_आकार = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ध्रुव वर्ण * cache_dir;
    std::unordered_set<ggml_backend_बफर_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_अवस्थाः;
};

शून्य rpc_server::hello(rpc_सन्देशः_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_संस्करणम्;
    response.minor = RPC_PROTO_MINOR_संस्करणम्;
    response.patch = RPC_PROTO_PATCH_संस्करणम्;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

तार्किक rpc_server::get_alloc_आकार(ध्रुव rpc_सन्देशः_get_alloc_आकार_req & request, rpc_सन्देशः_get_alloc_आकार_rsp & response) {
    uint32_t dev_id = request.device;
    यदि (dev_id >= backends.size()) {
        प्रत्यागम असत्य;
    }
    ggml_backend_बफर_type_t buft;
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    यदि (tensor == शून्यसूचकः) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        प्रत्यागम असत्य;
    }
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        यदि (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (शून्य*)tensor->बफर, tensor->दत्त);
    यदि (tensor->बफर == शून्यसूचकः) {
        //No buffer allocated.
        buft = ggml_backend_get_default_बफर_type(backends[dev_id]);
    } अन्यथा {
        buft = tensor->बफर->buft;
    }

    response.alloc_आकार = ggml_backend_buft_get_alloc_आकार(buft, tensor);

    प्रत्यागम सत्य;
}

तार्किक rpc_server::alloc_बफर(ध्रुव rpc_सन्देशः_alloc_बफर_req & request, rpc_सन्देशः_alloc_बफर_rsp & response) {
    uint32_t dev_id = request.device;
    यदि (dev_id >= backends.size()) {
        प्रत्यागम असत्य;
    }
    ggml_backend_बफर_type_t buft = ggml_backend_get_default_बफर_type(backends[dev_id]);
    ggml_backend_बफर_t बफर = ggml_backend_buft_alloc_बफर(buft, request.size);
    response.remote_ptr = 0;
    response.remote_आकार = 0;
    यदि (बफर != शून्यसूचकः) {
        response.remote_ptr = reinterpret_cast<uint64_t>(बफर);
        response.remote_आकार = बफर->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_आकार(request.size).c_str(), response.remote_ptr,
                rpc_log_format_आकार(response.remote_आकार).c_str());
        buffers.insert(बफर);
    } अन्यथा {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_आकार(request.size).c_str());
    }
    प्रत्यागम सत्य;
}

तार्किक rpc_server::get_alignment(ध्रुव rpc_सन्देशः_get_alignment_req & request, rpc_सन्देशः_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    यदि (dev_id >= backends.size()) {
        प्रत्यागम असत्य;
    }
    ggml_backend_बफर_type_t buft = ggml_backend_get_default_बफर_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_आकार(alignment).c_str());
    response.alignment = alignment;
    प्रत्यागम सत्य;
}

तार्किक rpc_server::get_max_आकार(ध्रुव rpc_सन्देशः_get_max_आकार_req & request, rpc_सन्देशः_get_max_आकार_rsp & response) {
    uint32_t dev_id = request.device;
    यदि (dev_id >= backends.size()) {
        प्रत्यागम असत्य;
    }
    ggml_backend_बफर_type_t buft = ggml_backend_get_default_बफर_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_आकार(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_आकार(max_size).c_str());
    response.max_size = max_size;
    प्रत्यागम सत्य;
}

तार्किक rpc_server::बफर_get_आधारः(ध्रुव rpc_सन्देशः_बफर_get_आधारः_req & request, rpc_सन्देशः_बफर_get_आधारः_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_बफर_t बफर = reinterpret_cast<ggml_backend_बफर_t>(request.remote_ptr);
    यदि (buffers.find(बफर) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        प्रत्यागम असत्य;
    }
    शून्य * आधारः = ggml_backend_बफर_get_आधारः(बफर);
    response.आधारः_ptr = reinterpret_cast<uint64_t>(आधारः);
    प्रत्यागम सत्य;
}

तार्किक rpc_server::free_बफर(ध्रुव rpc_सन्देशः_free_बफर_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_बफर_t बफर = reinterpret_cast<ggml_backend_बफर_t>(request.remote_ptr);
    यदि (buffers.find(बफर) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        प्रत्यागम असत्य;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    प्रति (स्वचालित & sgs : stored_graphs) {
        प्रति (स्वचालित & sg : sgs) {
            sg.second.graph = शून्यसूचकः;
        }
    }
    ggml_backend_बफर_free(बफर);
    buffers.erase(बफर);
    प्रत्यागम सत्य;
}

तार्किक rpc_server::बफर_clear(ध्रुव rpc_सन्देशः_बफर_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_बफर_t बफर = reinterpret_cast<ggml_backend_बफर_t>(request.remote_ptr);
    यदि (buffers.find(बफर) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        प्रत्यागम असत्य;
    }
    ggml_backend_बफर_clear(बफर, request.value);
    प्रत्यागम सत्य;
}

तार्किक rpc_server::memset_tensor(ध्रुव rpc_सन्देशः_memset_tensor_req & request) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    यदि (tensor == शून्यसूचकः || tensor->बफर == शून्यसूचकः) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        प्रत्यागम असत्य;
    }

    ध्रुव uint64_t tensor_आकार = ggml_nbytes(tensor);
    यदि (request.विस्थापनम् > tensor_आकार || request.size > tensor_आकार - request.विस्थापनम्) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.विस्थापनम्, request.size, tensor_आकार);
        प्रत्यागम असत्य;
    }

    ध्रुव uint64_t बफर_start = (uint64_t) ggml_backend_बफर_get_आधारः(tensor->बफर);
    ध्रुव uint64_t बफर_आकार = ggml_backend_बफर_get_आकार(tensor->बफर);
    यदि (request.tensor.दत्त < बफर_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        प्रत्यागम असत्य;
    }
    ध्रुव uint64_t दत्त_विस्थापनम् = request.tensor.दत्त - बफर_start;
    यदि (दत्त_विस्थापनम् > बफर_आकार ||
        request.विस्थापनम् > बफर_आकार - दत्त_विस्थापनम् ||
        request.size > बफर_आकार - दत्त_विस्थापनम् - request.विस्थापनम्) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        प्रत्यागम असत्य;
    }
    यदि (tensor->बफर->iface.memset_tensor == शून्यसूचकः) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        प्रत्यागम असत्य;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य *) tensor->बफर, tensor->दत्त, request.विस्थापनम्,
            rpc_log_format_आकार(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.विस्थापनम्, request.size);
    प्रत्यागम सत्य;
}

ggml_tensor * rpc_server::deserialize_tensor(संरचना ggml_context * ctx, ध्रुव rpc_tensor * tensor) {
    // Validate tensor type before using it
    यदि (tensor->type >= GGML_TYPE_गणनम्) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        प्रत्यागम शून्यसूचकः;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    यदि (ggml_blck_आकार((गणना ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        प्रत्यागम शून्यसूचकः;
    }

    ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    यदि (परिणाम == शून्यसूचकः) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        प्रत्यागम शून्यसूचकः;
    }

    प्रति (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम->nb[i] = tensor->nb[i];
    }
    परिणाम->बफर = reinterpret_cast<ggml_backend_बफर_t>(tensor->बफर);
    यदि (परिणाम->बफर && buffers.find(परिणाम->बफर) == buffers.end()) {
        परिणाम->बफर = शून्यसूचकः;
    }

    यदि (परिणाम->बफर && ggml_nelements(परिणाम) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_आकार = (uint64_t) ggml_nbytes(परिणाम);
        uint64_t बफर_start = (uint64_t) ggml_backend_बफर_get_आधारः(परिणाम->बफर);
        uint64_t बफर_आकार = (uint64_t) ggml_backend_बफर_get_आकार(परिणाम->बफर);
        यदि (tensor->दत्त + tensor_आकार < tensor->दत्त ||
            tensor->दत्त < बफर_start || tensor->दत्त + tensor_आकार > बफर_start + बफर_आकार) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(परिणाम->type),
                      परिणाम->ne[0], परिणाम->ne[1], परिणाम->ne[2], परिणाम->ne[3],
                      tensor->दत्त, tensor->दत्त + tensor_आकार, बफर_start, बफर_start + बफर_आकार);
            प्रत्यागम शून्यसूचकः;
        }
    }

    परिणाम->op = (ggml_op) tensor->op;
    प्रति (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / परिमाण(int32_t); i++) {
        परिणाम->op_params[i] = tensor->op_params[i];
    }
    परिणाम->flags = tensor->flags;
    परिणाम->दत्त = reinterpret_cast<शून्य *>(tensor->दत्त);
    ggml_समुच्चयः_name(परिणाम, tensor->name);
    प्रत्यागम परिणाम;
}


तार्किक rpc_server::समुच्चयः_tensor(ध्रुव std::vector<uint8_t> & निवेशः) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t विस्थापनम्;
    ध्रुव size_t शीर्षकम्_आकार = परिमाण(rpc_tensor) + परिमाण(cache_flag) + परिमाण(विस्थापनम्);
    यदि (निवेशः.size() < शीर्षकम्_आकार) {
        प्रत्यागम असत्य;
    }
    ध्रुव rpc_tensor * अन्तर्_tensor = (ध्रुव rpc_tensor *)निवेशः.दत्त();
    स्मृतिं_प्रतिलिख(&cache_flag, निवेशः.दत्त() + परिमाण(rpc_tensor), परिमाण(cache_flag));
    स्मृतिं_प्रतिलिख(&विस्थापनम्,     निवेशः.दत्त() + परिमाण(rpc_tensor) + परिमाण(cache_flag), परिमाण(विस्थापनम्));
    ध्रुव size_t size = निवेशः.size() - शीर्षकम्_आकार;

    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, अन्तर्_tensor);
    यदि (tensor == शून्यसूचकः || tensor->बफर == शून्यसूचकः) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        प्रत्यागम असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->बफर, tensor->दत्त, विस्थापनम्,
            rpc_log_format_आकार(size).c_str());

    // sanitize tensor->data
    {
        ध्रुव size_t p0 = (size_t) ggml_backend_बफर_get_आधारः(tensor->बफर);
        ध्रुव size_t p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        यदि (अन्तर्_tensor->दत्त + विस्थापनम् < p0 || अन्तर्_tensor->दत्त + विस्थापनम् >= p1 || size > (p1 - अन्तर्_tensor->दत्त - विस्थापनम्)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, अन्तर्_tensor->दत्त, विस्थापनम्, size, p0, p1);
            प्रत्यागम असत्य;
        }
    }

    ध्रुव शून्य * दत्त = निवेशः.दत्त() + शीर्षकम्_आकार;
    यदि (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ध्रुव uint8_t*)दत्त, size);
        वर्ण hash_str[17];
        snprintf(hash_str, परिमाण(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.लिख((ध्रुव वर्ण *)दत्त, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_समुच्चयः(tensor, दत्त, विस्थापनम्, size);
    प्रत्यागम सत्य;
}

तार्किक rpc_server::समुच्चयः_tensor_2d(ध्रुव std::vector<uint8_t> & निवेशः) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    यदि (निवेशः.size() < परिमाण(rpc_tensor) + 4*परिमाण(uint64_t)) {
        प्रत्यागम असत्य;
    }
    ध्रुव rpc_tensor * अन्तर्_tensor = (ध्रुव rpc_tensor *)निवेशः.दत्त();
    uint64_t शीर्षकम्[4];
    स्मृतिं_प्रतिलिख(शीर्षकम्, निवेशः.दत्त() + परिमाण(rpc_tensor), परिमाण(शीर्षकम्));
    ध्रुव uint64_t विस्थापनम्   = शीर्षकम्[0];
    ध्रुव uint64_t size     = शीर्षकम्[1];
    ध्रुव uint64_t n_copies = शीर्षकम्[2];
    ध्रुव uint64_t stride   = शीर्षकम्[3];

    ध्रुव uint64_t दत्त_आकार = निवेशः.size() - परिमाण(rpc_tensor) - 4*परिमाण(uint64_t);
    यदि (n_copies == 0 || size == 0 || size > दत्त_आकार / n_copies || size * n_copies != दत्त_आकार) {
        प्रत्यागम असत्य;
    }

    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, अन्तर्_tensor);
    यदि (tensor == शून्यसूचकः || tensor->बफर == शून्यसूचकः) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        प्रत्यागम असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->बफर, tensor->दत्त, विस्थापनम्,
            rpc_log_format_आकार(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        यदि (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            प्रत्यागम असत्य;
        }
        ध्रुव uint64_t विस्तृतिः = (n_copies - 1)*stride + size;
        ध्रुव uint64_t p0 = (uint64_t) ggml_backend_बफर_get_आधारः(tensor->बफर);
        ध्रुव uint64_t p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        यदि (अन्तर्_tensor->दत्त < p0 || अन्तर्_tensor->दत्त > p1 || विस्थापनम् > p1 - अन्तर्_tensor->दत्त || विस्तृतिः > p1 - अन्तर्_tensor->दत्त - विस्थापनम्) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, अन्तर्_tensor->दत्त, विस्थापनम्, विस्तृतिः, p0, p1);
            प्रत्यागम असत्य;
        }
        यदि (विस्थापनम् > ggml_nbytes(tensor) || विस्तृतिः > ggml_nbytes(tensor) - विस्थापनम्) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, विस्थापनम्, विस्तृतिः, ggml_nbytes(tensor));
            प्रत्यागम असत्य;
        }
    }

    ध्रुव शून्य * दत्त = निवेशः.दत्त() + परिमाण(rpc_tensor) + 4*परिमाण(uint64_t);
    ggml_backend_tensor_समुच्चयः_2d(tensor, दत्त, विस्थापनम्, size, n_copies, stride, size);
    प्रत्यागम सत्य;
}

तार्किक rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & दत्त) {
    यदि (!cache_dir) {
        प्रत्यागम असत्य;
    }
    वर्ण hash_str[17];
    snprintf(hash_str, परिमाण(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    यदि (!fs::exists(cache_file, ec)) {
        प्रत्यागम असत्य;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    दत्त.resize(size);
    ifs.पठ((वर्ण *)दत्त.दत्त(), size);
    प्रत्यागम सत्य;
}

तार्किक rpc_server::समुच्चयः_tensor_hash(ध्रुव rpc_सन्देशः_समुच्चयः_tensor_hash_req & request, rpc_सन्देशः_समुच्चयः_tensor_hash_rsp & response)
{
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    यदि (tensor == शून्यसूचकः || tensor->बफर == शून्यसूचकः) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        प्रत्यागम असत्य;
    }

    std::vector<uint8_t> cached_file;
    यदि (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.विस्थापनम्, request.hash);
        response.परिणाम = 0;
        प्रत्यागम सत्य;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->बफर, tensor->दत्त, request.विस्थापनम्,
            rpc_log_format_आकार(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ध्रुव size_t p0 = (size_t) ggml_backend_बफर_get_आधारः(tensor->बफर);
        ध्रुव size_t p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        यदि (request.tensor.दत्त + request.विस्थापनम् < p0
         || request.tensor.दत्त + request.विस्थापनम् >= p1
         || size > (p1 - request.tensor.दत्त - request.विस्थापनम्)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.दत्त, request.विस्थापनम्, size, request.hash, p0, p1);
            प्रत्यागम असत्य;
        }
    }
    ggml_backend_tensor_समुच्चयः(tensor, cached_file.दत्त(), request.विस्थापनम्, size);
    response.परिणाम = 1;
    प्रत्यागम सत्य;
}

तार्किक rpc_server::init_tensor(ध्रुव rpc_सन्देशः_init_tensor_req & request) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    यदि (tensor == शून्यसूचकः) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        प्रत्यागम असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (शून्य*)tensor->बफर, tensor->दत्त);
    // Call the backend's buffer_init_tensor function
    ggml_backend_बफर_t बफर = tensor->बफर;
    यदि (बफर && बफर->iface.init_tensor) {
        बफर->iface.init_tensor(बफर, tensor);
    } अन्यथा {
        यदि (!बफर) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    यदि (tensor->extra != शून्यसूचकः) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

तार्किक rpc_server::get_tensor(ध्रुव rpc_सन्देशः_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    यदि (tensor == शून्यसूचकः || tensor->बफर == शून्यसूचकः) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        प्रत्यागम असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->बफर, tensor->दत्त, request.विस्थापनम्,
             rpc_log_format_आकार(request.size).c_str());

    // sanitize tensor->data
    {
        ध्रुव size_t p0 = (size_t) ggml_backend_बफर_get_आधारः(tensor->बफर);
        ध्रुव size_t p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        यदि (request.tensor.दत्त + request.विस्थापनम् < p0 ||
            request.tensor.दत्त + request.विस्थापनम् >= p1 ||
            request.size > (p1 - request.tensor.दत्त - request.विस्थापनम्)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.दत्त, request.विस्थापनम्, request.size, p0, p1);
                प्रत्यागम असत्य;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.दत्त(), request.विस्थापनम्, request.size);
    प्रत्यागम सत्य;
}

तार्किक rpc_server::get_tensor_2d(ध्रुव rpc_सन्देशः_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    यदि (tensor == शून्यसूचकः || tensor->बफर == शून्यसूचकः) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        प्रत्यागम असत्य;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (शून्य*)tensor->बफर, tensor->दत्त, request.विस्थापनम्,
            rpc_log_format_आकार(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        यदि (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            प्रत्यागम असत्य;
        }
        यदि (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            प्रत्यागम असत्य;
        }
        ध्रुव uint64_t विस्तृतिः = (request.n_copies - 1)*request.stride + request.size;
        ध्रुव uint64_t p0 = (uint64_t) ggml_backend_बफर_get_आधारः(tensor->बफर);
        ध्रुव uint64_t p1 = p0 + ggml_backend_बफर_get_आकार(tensor->बफर);

        यदि (request.tensor.दत्त < p0 || request.tensor.दत्त > p1 || request.विस्थापनम् > p1 - request.tensor.दत्त ||
                विस्तृतिः > p1 - request.tensor.दत्त - request.विस्थापनम्) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.दत्त, request.विस्थापनम्, विस्तृतिः, p0, p1);
            प्रत्यागम असत्य;
        }
        यदि (request.विस्थापनम् > ggml_nbytes(tensor) || विस्तृतिः > ggml_nbytes(tensor) - request.विस्थापनम्) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.विस्थापनम्, विस्तृतिः, ggml_nbytes(tensor));
            प्रत्यागम असत्य;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.दत्त(), request.विस्थापनम्, request.size, request.n_copies, request.stride, request.size);
    प्रत्यागम सत्य;
}

तार्किक rpc_server::copy_tensor(ध्रुव rpc_सन्देशः_copy_tensor_req & request, rpc_सन्देशः_copy_tensor_rsp & response) {
    sync_all_backends();
    संरचना ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    यदि (src == शून्यसूचकः || dst == शून्यसूचकः || src->बफर == शून्यसूचकः || dst->बफर == शून्यसूचकः) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        प्रत्यागम असत्य;
    }

    uint64_t src_आकार   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_दत्त   = (uint64_t) dst->दत्त;
    uint64_t dst_आधारः   = (uint64_t) ggml_backend_बफर_get_आधारः(dst->बफर);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_बफर_get_आकार(dst->बफर);

    यदि (dst_दत्त + src_आकार > dst_आधारः + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_दत्त,
                         dst_दत्त + src_आकार,
                         dst_आधारः,
                         dst_आधारः + dst_buf_sz);
        प्रत्यागम असत्य;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.परिणाम = ggml_backend_बफर_copy_tensor(src, dst);
    प्रत्यागम सत्य;
}

ggml_tensor * rpc_server::create_ग्रन्थिः(uint64_t id,
                                      संरचना ggml_context * ctx,
                                      ध्रुव std::unordered_map<uint64_t, ध्रुव rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, संरचना ggml_tensor*> & tensor_map) {
    यदि (tensor_map.find(id) != tensor_map.end()) {
        प्रत्यागम tensor_map[id];
    }
    // Safely find the tensor pointer
    स्वचालित it_ptr = tensor_ptrs.find(id);
    यदि (it_ptr == tensor_ptrs.end()) {
        प्रत्यागम शून्यसूचकः;
    }
    ध्रुव rpc_tensor * tensor = it_ptr->second;

    संरचना ggml_tensor * परिणाम = deserialize_tensor(ctx, tensor);
    यदि (परिणाम == शून्यसूचकः) {
        प्रत्यागम शून्यसूचकः;
    }
    यदि (परिणाम->बफर == शून्यसूचकः && परिणाम->दत्त != शून्यसूचकः) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        प्रत्यागम शून्यसूचकः;
    }
    tensor_map[id] = परिणाम;
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        यदि (tensor->src[i] == 0) {
            परिणाम->src[i] = शून्यसूचकः;
        } अन्यथा {
            परिणाम->src[i] = create_ग्रन्थिः(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            यदि (परिणाम->src[i] == शून्यसूचकः) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                प्रत्यागम शून्यसूचकः;
            }
        }
    }

    // Handle view_src similarly
    यदि (tensor->view_src == 0) {
        परिणाम->view_src = शून्यसूचकः;
    } अन्यथा {
        परिणाम->view_src = create_ग्रन्थिः(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        यदि (परिणाम->view_src == शून्यसूचकः) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            प्रत्यागम शून्यसूचकः;
        }
    }
    परिणाम->view_offs = tensor->view_offs;
    प्रत्यागम परिणाम;
}

तार्किक rpc_server::graph_compute(ध्रुव std::vector<uint8_t> & निवेशः) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    यदि (निवेशः.size() < 2*परिमाण(uint32_t) + परिमाण(uint64_t)) {
        प्रत्यागम असत्य;
    }
    ध्रुव uint8_t * src = निवेशः.दत्त();
    uint32_t device;
    स्मृतिं_प्रतिलिख(&device, src, परिमाण(device));
    src += परिमाण(device);
    यदि (device >= backends.size()) {
        प्रत्यागम असत्य;
    }
    uint64_t uid;
    स्मृतिं_प्रतिलिख(&uid, src, परिमाण(uid));
    src += परिमाण(uid);
    uint32_t n_nodes;
    स्मृतिं_प्रतिलिख(&n_nodes, src, परिमाण(n_nodes));
    src += परिमाण(n_nodes);
    यदि (निवेशः.size() < 2*परिमाण(uint32_t) + परिमाण(uint64_t) + n_nodes*परिमाण(uint64_t) + परिमाण(uint32_t)) {
        प्रत्यागम असत्य;
    }
    ध्रुव uint64_t * nodes = (ध्रुव uint64_t *)src;
    src += n_nodes*परिमाण(uint64_t);
    uint32_t n_tensors;
    स्मृतिं_प्रतिलिख(&n_tensors, src, परिमाण(n_tensors));
    src += परिमाण(n_tensors);
    यदि (निवेशः.size() < 2*परिमाण(uint32_t) + परिमाण(uint64_t) + n_nodes*परिमाण(uint64_t) + परिमाण(uint32_t) + n_tensors*परिमाण(rpc_tensor)) {
        प्रत्यागम असत्य;
    }
    ध्रुव rpc_tensor * tensors = (ध्रुव rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    यदि (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_आकार = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, असत्य);
    यदि (sg.बफर.size() < buf_आकार) {
        sg.बफर.resize(buf_आकार);
    }
    संरचना ggml_init_params params = {
        /*.mem_size   =*/ buf_आकार,
        /*.mem_buffer =*/ sg.बफर.दत्त(),
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    संरचना ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, असत्य);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ध्रुव rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    प्रति (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ध्रुव स्वचालित t_deserialize = std::chrono::steady_clock::now();
    प्रति (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        स्मृतिं_प्रतिलिख(&id, &nodes[i], परिमाण(id));
        graph->nodes[i] = create_ग्रन्थिः(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        यदि (graph->nodes[i] == शून्यसूचकः && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            प्रत्यागम असत्य;
        }
        यदि (graph->nodes[i] != शून्यसूचकः) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ध्रुव size_t hash_pos = ggml_hash_निवेशय(&graph->visited_hash_समुच्चयः, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_गणनम्;
        }
    }
    ध्रुव std::chrono::duration<द्विगुण, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_स्थितिः स्थितिः = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(स्थितिः == GGML_स्थितिः_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    प्रत्यागम सत्य;
}

तार्किक rpc_server::graph_recompute(ध्रुव rpc_सन्देशः_graph_recompute_req & request) {
    uint32_t device = request.device;
    यदि (device >= backends.size()) {
        प्रत्यागम असत्य;
    }
    स्वचालित it = stored_graphs[device].find(request.uid);
    यदि (it == stored_graphs[device].end() || it->second.graph == शून्यसूचकः) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        प्रत्यागम असत्य;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_स्थितिः स्थितिः = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(स्थितिः == GGML_स्थितिः_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    प्रत्यागम सत्य;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
शून्य rpc_server::sync_all_backends() {
    प्रति (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
तार्किक rpc_server::comm_init(ध्रुव rpc_सन्देशः_comm_init_req & request, rpc_सन्देशः_comm_init_rsp & response) {
    response.ok = 0;
    यदि (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        प्रत्यागम सत्य;
    }
    comm_state & state = comm_अवस्थाः[request.device];
    यदि (state.peer != शून्यसूचकः) {
        response.ok = 1;
        प्रत्यागम सत्य;
    }
    uint8_t local_caps[RPC_CONN_CAPS_आकार] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_आकार] = {};
    यदि (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.पत्तनम्);
        यदि (srv == शून्यसूचकः) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.पत्तनम्);
            प्रत्यागम सत्य;
        }
        state.peer = srv->accept();
        यदि (state.peer == शून्यसूचकः) {
            प्रत्यागम सत्य;
        }
        यदि (!state.peer->recv_दत्त(remote_caps, परिमाण(remote_caps))) {
            state.peer = शून्यसूचकः;
            प्रत्यागम सत्य;
        }
        state.peer->get_caps(local_caps);
        यदि (!state.peer->send_दत्त(local_caps, परिमाण(local_caps))) {
            state.peer = शून्यसूचकः;
            प्रत्यागम सत्य;
        }
        state.peer->update_caps(remote_caps);
    } अन्यथा {
        ध्रुव std::string host(request.host, strnlen(request.host, परिमाण(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        प्रति (पूर्णाङ्क i = 0; i < 100 && state.peer == शून्यसूचकः; i++) {
            state.peer = socket_t::connect(host.c_str(), request.पत्तनम्);
            यदि (state.peer == शून्यसूचकः) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        यदि (state.peer == शून्यसूचकः) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.पत्तनम्);
            प्रत्यागम सत्य;
        }
        state.peer->get_caps(local_caps);
        यदि (!state.peer->send_दत्त(local_caps, परिमाण(local_caps)) ||
            !state.peer->recv_दत्त(remote_caps, परिमाण(remote_caps))) {
            state.peer = शून्यसूचकः;
            प्रत्यागम सत्य;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    प्रत्यागम सत्य;
}

तार्किक rpc_server::comm_allreduce(ध्रुव rpc_सन्देशः_comm_allreduce_req & request) {
    यदि (request.device >= backends.size()) {
        प्रत्यागम असत्य;
    }
    comm_state & state = comm_अवस्थाः[request.device];
    यदि (state.peer == शून्यसूचकः) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        प्रत्यागम असत्य;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_आकार = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, असत्य);
    संरचना ggml_init_params params = {
        /*.mem_size   =*/ ctx_आकार,
        /*.mem_buffer =*/ रिक्त,
        /*.no_alloc   =*/ सत्य,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != शून्यसूचकः);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    यदि (t_dst == शून्यसूचकः || t_dst->बफर == शून्यसूचकः) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        प्रत्यागम असत्य;
    }
    ध्रुव size_t  nbytes = ggml_nbytes(t_dst);
    ध्रुव int64_t ne     = ggml_nelements(t_dst);
    यदि (nbytes == 0) {
        प्रत्यागम सत्य;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ध्रुव तार्किक   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ध्रुव size_t wire_अष्टकानि = wire_bf16 ? (size_t) ne*2 : nbytes;
    ध्रुव size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    यदि (state.scratch_आकार < need) {
        state.scratch.reset(ggml_backend_alloc_बफर(backend, need));
        state.scratch_आकार = need;
    }
    वर्ण * scratch_आधारः = (वर्ण *) ggml_backend_बफर_get_आधारः(state.scratch.get());
    state.send_buf.resize(wire_अष्टकानि);
    state.recv_buf.resize(wire_अष्टकानि);

    स्वचालित new_scratch_tensor = [&](ggml_type type, size_t विस्थापनम्) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->बफर = state.scratch.get();
        t->दत्त   = scratch_आधारः + विस्थापनम्;
        प्रत्यागम t;
    };
    स्वचालित new_cpy_ग्रन्थिः = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->बफर = dst->बफर;
        t->दत्त   = dst->दत्त;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        प्रत्यागम t;
    };
    स्वचालित compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, असत्य);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != शून्यसूचकः ? 2 : 1;
        ggml_स्थितिः स्थितिः = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(स्थितिः == GGML_स्थितिः_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = शून्यसूचकः;
    ggml_tensor * t_wire_recv = शून्यसूचकः;
    यदि (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_ग्रन्थिः(t_dst, t_wire_send), शून्यसूचकः);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.दत्त(), 0, wire_अष्टकानि);
    } अन्यथा {
        ggml_backend_tensor_get(t_dst, state.send_buf.दत्त(), 0, wire_अष्टकानि);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    यदि (state.rank == 0) {
        यदि (!state.peer->send_दत्त(state.send_buf.दत्त(), wire_अष्टकानि) || !state.peer->flush() ||
            !state.peer->recv_दत्त(state.recv_buf.दत्त(), wire_अष्टकानि)) {
            प्रत्यागम असत्य;
        }
    } अन्यथा {
        यदि (!state.peer->recv_दत्त(state.recv_buf.दत्त(), wire_अष्टकानि) ||
            !state.peer->send_दत्त(state.send_buf.दत्त(), wire_अष्टकानि) || !state.peer->flush()) {
            प्रत्यागम असत्य;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = शून्यसूचकः;
    यदि (wire_bf16) {
        ggml_backend_tensor_समुच्चयः(t_wire_recv, state.recv_buf.दत्त(), 0, wire_अष्टकानि);
        t_cast = new_cpy_ग्रन्थिः(t_wire_recv, t_peer);
    } अन्यथा {
        ggml_backend_tensor_समुच्चयः(t_peer, state.recv_buf.दत्त(), 0, wire_अष्टकानि);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_सङ्कलय;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->बफर = t_dst->बफर;
    t_red->दत्त   = t_dst->दत्त;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    यदि (t_cast != शून्यसूचकः) {
        compute_nodes(t_cast, t_red);
    } अन्यथा {
        compute_nodes(t_red, शून्यसूचकः);
    }
    प्रत्यागम सत्य;
}

तार्किक rpc_server::comm_free(ध्रुव rpc_सन्देशः_comm_free_req & request) {
    यदि (request.device >= backends.size()) {
        प्रत्यागम असत्य;
    }
    comm_अवस्थाः[request.device] = comm_state();
    प्रत्यागम सत्य;
}

तार्किक rpc_server::get_device_स्मृतिः(ध्रुव rpc_सन्देशः_get_device_स्मृतिः_req & request, rpc_सन्देशः_get_device_स्मृतिः_rsp & response) {
    uint32_t dev_id = request.device;
    यदि (dev_id >= backends.size()) {
        प्रत्यागम असत्य;
    }
    size_t मुक्त, योग;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_स्मृतिः(dev, &मुक्त, &योग);
    response.free_mem = मुक्त;
    response.योग_mem = योग;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_आकार(response.free_mem).c_str(), rpc_log_format_आकार(response.योग_mem).c_str());
    प्रत्यागम सत्य;
}

rpc_server::~rpc_server() {
    प्रति (स्वचालित बफर : buffers) {
        ggml_backend_बफर_free(बफर);
    }
}

स्थिर शून्य rpc_serve_client(ध्रुव std::vector<ggml_backend_t> & backends, ध्रुव वर्ण * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    यदि (!sock->recv_दत्त(&cmd, 1)) {
        प्रत्यागम;
    }
    यदि (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        प्रत्यागम;
    }

    // Read input_size and validate protocol version
    uint64_t hello_निवेशः_आकार;
    यदि (!sock->recv_दत्त(&hello_निवेशः_आकार, परिमाण(hello_निवेशः_आकार))) {
        प्रत्यागम;
    }

    यदि (hello_निवेशः_आकार != परिमाण(rpc_सन्देशः_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_निवेशः_आकार, परिमाण(rpc_सन्देशः_hello_req), RPC_PROTO_MAJOR_संस्करणम्);
        प्रत्यागम;
    }

    rpc_सन्देशः_hello_req req = {};
    यदि (!sock->recv_दत्त(&req, परिमाण(req))) {
        प्रत्यागम;
    }

    rpc_सन्देशः_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    यदि (!send_सन्देशः(sock, &rsp, परिमाण(rsp))) {
        प्रत्यागम;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    यावत् (सत्य) {
        यदि (!sock->recv_दत्त(&cmd, 1)) {
            भङ्ग;
        }
        यदि (cmd >= RPC_CMD_गणनम्) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            भङ्ग;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((गणना rpc_cmd)cmd));
        विकल्प (cmd) {
            पक्ष RPC_CMD_HELLO: {
                // HELLO command is handled above
                प्रत्यागम;
            }
            पक्ष RPC_CMD_DEVICE_गणनम्: {
                यदि (!recv_सन्देशः(sock, शून्यसूचकः, 0)) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_device_गणनम्_rsp response;
                response.device_गणनम् = backends.size();
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_ALLOC_बफर: {
                rpc_सन्देशः_alloc_बफर_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_alloc_बफर_rsp response;
                यदि (!server.alloc_बफर(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_GET_ALLOC_आकार: {
                rpc_सन्देशः_get_alloc_आकार_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_get_alloc_आकार_rsp response;
                यदि (!server.get_alloc_आकार(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_GET_ALIGNMENT: {
                rpc_सन्देशः_get_alignment_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_get_alignment_rsp response;
                यदि (!server.get_alignment(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_GET_MAX_आकार: {
                rpc_सन्देशः_get_max_आकार_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_get_max_आकार_rsp response;
                यदि (!server.get_max_आकार(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_बफर_GET_आधारः: {
                rpc_सन्देशः_बफर_get_आधारः_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_बफर_get_आधारः_rsp response;
                यदि (!server.बफर_get_आधारः(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_FREE_बफर: {
                rpc_सन्देशः_free_बफर_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                यदि (!server.free_बफर(request)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_बफर_CLEAR: {
                rpc_सन्देशः_बफर_clear_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                यदि (!server.बफर_clear(request)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_MEMSET_TENSOR: {
                rpc_सन्देशः_memset_tensor_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                यदि (!server.memset_tensor(request)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_समुच्चयः_TENSOR: {
                std::vector<uint8_t> निवेशः;
                यदि (!recv_सन्देशः(sock, निवेशः)) {
                    प्रत्यागम;
                }
                यदि (!server.समुच्चयः_tensor(निवेशः)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_समुच्चयः_TENSOR_2_D: {
                std::vector<uint8_t> निवेशः;
                यदि (!recv_सन्देशः(sock, निवेशः)) {
                    प्रत्यागम;
                }
                यदि (!server.समुच्चयः_tensor_2d(निवेशः)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_GET_TENSOR_2D: {
                rpc_सन्देशः_get_tensor_2d_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                std::vector<uint8_t> response;
                यदि (!server.get_tensor_2d(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, response.दत्त(), response.size())) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_समुच्चयः_TENSOR_HASH: {
                rpc_सन्देशः_समुच्चयः_tensor_hash_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_समुच्चयः_tensor_hash_rsp response;
                यदि (!server.समुच्चयः_tensor_hash(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_INIT_TENSOR: {
                rpc_सन्देशः_init_tensor_req request;
                यदि (!recv_सन्देशः(sock, &request,परिमाण(request))) {
                    प्रत्यागम;
                }
                यदि (!server.init_tensor(request)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_GET_TENSOR: {
                rpc_सन्देशः_get_tensor_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                std::vector<uint8_t> response;
                यदि (!server.get_tensor(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, response.दत्त(), response.size())) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_COPY_TENSOR: {
                rpc_सन्देशः_copy_tensor_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_copy_tensor_rsp response;
                यदि (!server.copy_tensor(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> निवेशः;
                यदि (!recv_सन्देशः(sock, निवेशः)) {
                    प्रत्यागम;
                }
                यदि (!server.graph_compute(निवेशः)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_सन्देशः_graph_recompute_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                यदि (!server.graph_recompute(request)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_COMM_INIT: {
                rpc_सन्देशः_comm_init_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_comm_init_rsp response;
                यदि (!server.comm_init(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_COMM_ALLREDUCE: {
                rpc_सन्देशः_comm_allreduce_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                यदि (!server.comm_allreduce(request)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_COMM_FREE: {
                rpc_सन्देशः_comm_free_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                यदि (!server.comm_free(request)) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            पक्ष RPC_CMD_GET_DEVICE_स्मृतिः: {
                rpc_सन्देशः_get_device_स्मृतिः_req request;
                यदि (!recv_सन्देशः(sock, &request, परिमाण(request))) {
                    प्रत्यागम;
                }
                rpc_सन्देशः_get_device_स्मृतिः_rsp response;
                यदि (!server.get_device_स्मृतिः(request, response)) {
                    प्रत्यागम;
                }
                यदि (!send_सन्देशः(sock, &response, परिमाण(response))) {
                    प्रत्यागम;
                }
                भङ्ग;
            }
            स्वभाव: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((गणना rpc_cmd)cmd));
                प्रत्यागम;
            }
        }
    }
}

शून्य ggml_backend_rpc_start_server(ध्रुव वर्ण * endpoint, ध्रुव वर्ण * cache_dir,
                                   size_t n_तन्तवः, size_t n_devices, ggml_backend_dev_t * devices) {
    यदि (n_devices == 0 || devices == शून्यसूचकः) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        प्रत्यागम;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_संस्करणम्,
        RPC_PROTO_MINOR_संस्करणम्,
        RPC_PROTO_PATCH_संस्करणम्);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    प्रति (size_t i = 0; i < n_devices; i++) {
        स्वचालित dev = devices[i];
        size_t मुक्त, योग;
        ggml_backend_dev_स्मृतिः(dev, &मुक्त, &योग);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 योग / 1024 / 1024, मुक्त / 1024 / 1024);
        स्वचालित backend = ggml_backend_dev_init(dev, शून्यसूचकः);
        यदि (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            प्रत्यागम;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : शून्यसूचकः;
        यदि (reg) {
            स्वचालित ggml_backend_समुच्चयः_n_तन्तवः_fn = (ggml_backend_समुच्चयः_n_तन्तवः_t) ggml_backend_reg_get_proc_स्थानाङ्कः(reg, "ggml_backend_set_n_threads");
            यदि (ggml_backend_समुच्चयः_n_तन्तवः_fn) {
                ggml_backend_समुच्चयः_n_तन्तवः_fn(backend, n_तन्तवः);
            }
        }
    }

    std::string host;
    पूर्णाङ्क पत्तनम्;
    यदि (!parse_endpoint(endpoint, host, पत्तनम्)) {
        प्रत्यागम;
    }

#यदि_परिभाषितम् GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#अन्यथा
    LOG_INFO("  transport      : TCP\n");
#यदि_अन्तः // GGML_RPC_RDMA
    यदि (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        प्रत्यागम;
    }
    स्वचालित server_socket = socket_t::create_server(host.c_str(), पत्तनम्);
    यदि (server_socket == शून्यसूचकः) {
        LOG_ERROR("Failed to create server socket\n");
        प्रत्यागम;
    }
    यावत् (सत्य) {
        स्वचालित client_socket = server_socket->accept();
        यदि (client_socket == शून्यसूचकः) {
            LOG_ERROR("Failed to accept client connection\n");
            प्रत्यागम;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    प्रति (स्वचालित backend : backends) {
        ggml_backend_free(backend);
    }
}

स्थिर ध्रुव वर्ण * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    प्रत्यागम ctx->name.c_str();
}

स्थिर ध्रुव वर्ण * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    प्रत्यागम ctx->description.c_str();
}

स्थिर शून्य ggml_backend_rpc_device_get_स्मृतिः(ggml_backend_dev_t dev, size_t * मुक्त, size_t * योग) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_स्मृतिः(ctx->endpoint.c_str(), ctx->device, मुक्त, योग);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर शून्य ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);
    props->caps = {
        /* .async                 = */ सत्य,
        /* .host_buffer           = */ असत्य,
        /* .buffer_from_host_ptr  = */ असत्य,
        /* .events                = */ सत्य,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    प्रत्यागम ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_अप्रयुक्तम्(params);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_rpc_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    प्रत्यागम ggml_backend_rpc_बफर_type(ctx->endpoint.c_str(), ctx->device);

    GGML_अप्रयुक्तम्(dev);
}

स्थिर तार्किक ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ध्रुव संरचना ggml_tensor * op) {
    GGML_अप्रयुक्तम्(dev);
    GGML_अप्रयुक्तम्(op);
    //TODO: call the remote backend and cache the results
    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    यदि (!buft || buft->iface.get_name != ggml_backend_rpc_बफर_type_name) {
        प्रत्यागम असत्य;
    }
    ggml_backend_rpc_बफर_type_context * buft_ctx = (ggml_backend_rpc_बफर_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    प्रत्यागम buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

स्थिर ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वचालित dispatcher = get_dispatcher(ctx->endpoint);
    प्रत्यागम dispatcher->event_new(dev);
}

स्थिर शून्य ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वचालित dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

स्थिर शून्य ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    स्वचालित dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

स्थिर ध्रुव संरचना ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_स्मृतिः,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_बफर_type,
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

स्थिर ध्रुव वर्ण * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    प्रत्यागम ctx ? ctx->name.c_str() : "RPC";
}

स्थिर size_t ggml_backend_rpc_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    प्रत्यागम ctx ? ctx->devices.size() : 0;
}

स्थिर ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t सूचकांक) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    यदि (ctx == शून्यसूचकः) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } अन्यथा {
        GGML_ASSERT(सूचकांक < ctx->devices.size());
        प्रत्यागम ctx->devices[सूचकांक];
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
    यदि (comm_ctx == शून्यसूचकः) {
        प्रत्यागम;
    }
    प्रति (ध्रुव स्वचालित & rank : comm_ctx->ranks) {
        स्वचालित request = std::make_shared<rpc_सन्देशः_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, परिमाण(*request));
        rank.dispatcher->busy_spin_release();
    }
    विलोपय comm_ctx;
}

स्थिर शून्य * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    यदि (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != शून्यसूचकः) {
        प्रत्यागम शून्यसूचकः;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    प्रति (size_t i = 0; i < n_backends; i++) {
        यदि (!ggml_backend_is_rpc(backends[i])) {
            प्रत्यागम शून्यसूचकः;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        प्रति (ध्रुव स्वचालित & rank : ranks) {
            यदि (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                प्रत्यागम शून्यसूचकः;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    पूर्णाङ्क port0;
    यदि (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        प्रत्यागम शून्यसूचकः;
    }
    ध्रुव uint32_t comm_पत्तनम् = (uint32_t) port0 + 1000;
    यदि (host0.size() >= 64) {
        प्रत्यागम शून्यसूचकः;
    }

    प्रति (ध्रुव स्वचालित & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_सन्देशः_comm_init_rsp> responses(n_backends);
    प्रति (size_t i = 0; i < n_backends; i++) {
        स्वचालित request = std::make_shared<rpc_सन्देशः_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->पत्तनम्   = comm_पत्तनम्;
        यदि (i > 0) {
            स्मृतिं_प्रतिलिख(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, परिमाण(*request), &responses[i], परिमाण(responses[i]));
    }
    प्रति (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    तार्किक ok = सत्य;
    प्रति (size_t i = 0; i < n_backends; i++) {
        यदि (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = असत्य;
        }
    }
    यदि (!ok) {
        प्रति (ध्रुव स्वचालित & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        प्रत्यागम शून्यसूचकः;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    प्रत्यागम नूतन ggml_backend_rpc_comm_context{std::move(ranks)};
}

स्थिर तार्किक ggml_backend_rpc_comm_allreduce_tensor(शून्य * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    यदि (comm_ctx == शून्यसूचकः) {
        प्रत्यागम असत्य;
    }
    ध्रुव size_t n_ranks = comm_ctx->ranks.size();
    ध्रुव int64_t ne = ggml_nelements(tensors[0]);
    यदि (ne == 0) {
        प्रत्यागम सत्य;
    }
    प्रति (size_t i = 0; i < n_ranks; i++) {
        यदि (tensors[i] == शून्यसूचकः || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->बफर == शून्यसूचकः || !ggml_backend_बफर_is_rpc(tensors[i]->बफर)) {
            प्रत्यागम असत्य;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        यदि ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            प्रत्यागम असत्य;
        }
    }
    प्रति (size_t i = 0; i < n_ranks; i++) {
        स्वचालित request = std::make_shared<rpc_सन्देशः_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, परिमाण(*request));
    }
    प्रत्यागम सत्य;
}

स्थिर शून्य * ggml_backend_rpc_get_proc_स्थानाङ्कः(ggml_backend_reg_t reg, ध्रुव वर्ण * name) {
    यदि (std::वाक्यं_तुलय(name, "ggml_backend_rpc_add_server") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_rpc_सङ्कलय_server;
    }
    यदि (std::वाक्यं_तुलय(name, "ggml_backend_rpc_start_server") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_rpc_start_server;
    }
    यदि (std::वाक्यं_तुलय(name, "ggml_backend_comm_init") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_rpc_comm_init;
    }
    यदि (std::वाक्यं_तुलय(name, "ggml_backend_comm_free") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_rpc_comm_free;
    }
    यदि (std::वाक्यं_तुलय(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    प्रत्यागम रिक्त;

    GGML_अप्रयुक्तम्(reg);
}

स्थिर ध्रुव संरचना ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_गणनम्,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_स्थानाङ्कः,
};

ggml_backend_reg_t ggml_backend_rpc_reg(शून्य) {
    स्थिर संरचना ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_संस्करणम्,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ रिक्त,
    };

    प्रत्यागम &ggml_backend_rpc_reg;
}

स्थिर uint32_t ggml_backend_rpc_get_device_गणनम्(ध्रुव वर्ण * endpoint) {
    स्वचालित dispatcher = get_dispatcher(endpoint);
    rpc_सन्देशः_device_गणनम्_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_गणनम्, शून्यसूचकः, 0, &response, परिमाण(response));
    प्रत्यागम response.device_गणनम्;
}

स्थिर ध्रुव ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_गणनम्,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_स्थानाङ्कः,
};

ggml_backend_reg_t ggml_backend_rpc_सङ्कलय_server(ध्रुव वर्ण * endpoint) {
    स्थिर std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    स्थिर std::mutex mutex;
    स्थिर uint32_t dev_id = 0;
    std::lock_guard<std::mutex> तालकम्(mutex);
    यदि (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        प्रत्यागम reg_map[endpoint];
    }
    uint32_t dev_गणनम् = ggml_backend_rpc_get_device_गणनम्(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_गणनम्);
    यदि (dev_गणनम् == 0) {
        प्रत्यागम शून्यसूचकः;
    }
    ggml_backend_rpc_reg_context * ctx = नूतन ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    प्रति (uint32_t ind = 0; ind < dev_गणनम्; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = नूतन ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = नूतन ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = नूतन ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_संस्करणम्,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    प्रत्यागम reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

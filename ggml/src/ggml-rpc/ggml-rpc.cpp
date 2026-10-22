#ضمن "ggml-rpc.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpp.h"
#ضمن "transport.h"
#ضمن "log.h"

#ضمن <array>
#ضمن <chrono>
#ضمن <cinttypes>
#ضمن <optional>
#ضمن <string>
#ضمن <thread>
#ضمن <vector>
#ضمن <queue>
#ضمن <condition_variable>
#ضمن <future>
#ضمن <memory>
#ضمن <mutex>
#ضمن <unordered_map>
#ضمن <unordered_set>
#ضمن <cstring>
#ضمن <fstream>
#ضمن <filesystem>
#ضمن <algorithm>
#ضمن <atomic>

نطاق fs = std::filesystem;

// macro for nicer error messages on server crash
#وصف_ڪريو RPC_حالت_ASSERT(x) إذا (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#هدايت pack(ڌڪو, 1)
// ggml_tensor is serialized into rpc_tensor
بنية rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t بفر;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / حجم(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t معلومات;
    حرف name[GGML_MAX_NAME];

    int32_t use_ڳڻپ;
};

static_assert(حجم(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
تعداد rpc_cmd {
    RPC_CMD_ALLOC_بفر = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_سائيز,
    RPC_CMD_بفر_GET_بنياد,
    RPC_CMD_FREE_بفر,
    RPC_CMD_بفر_CLEAR,
    RPC_CMD_سيٽ_TENSOR,
    RPC_CMD_سيٽ_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_گراف_COMPUTE,
    RPC_CMD_GET_DEVICE_يادگيري,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_سائيز,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_ڳڻپ,
    RPC_CMD_گراف_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_سيٽ_TENSOR_2_D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_ڳڻپ,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

ساكن ثابت حرف * rpc_cmd_name(تعداد rpc_cmd cmd) {
    بدل (cmd) {
        حالة RPC_CMD_ALLOC_بفر:      أرجع "ALLOC_BUFFER";
        حالة RPC_CMD_GET_ALIGNMENT:     أرجع "GET_ALIGNMENT";
        حالة RPC_CMD_GET_MAX_سائيز:      أرجع "GET_MAX_SIZE";
        حالة RPC_CMD_بفر_GET_بنياد:   أرجع "BUFFER_GET_BASE";
        حالة RPC_CMD_FREE_بفر:       أرجع "FREE_BUFFER";
        حالة RPC_CMD_بفر_CLEAR:      أرجع "BUFFER_CLEAR";
        حالة RPC_CMD_سيٽ_TENSOR:        أرجع "SET_TENSOR";
        حالة RPC_CMD_سيٽ_TENSOR_HASH:   أرجع "SET_TENSOR_HASH";
        حالة RPC_CMD_GET_TENSOR:        أرجع "GET_TENSOR";
        حالة RPC_CMD_COPY_TENSOR:       أرجع "COPY_TENSOR";
        حالة RPC_CMD_گراف_COMPUTE:     أرجع "GRAPH_COMPUTE";
        حالة RPC_CMD_GET_DEVICE_يادگيري: أرجع "GET_DEVICE_MEMORY";
        حالة RPC_CMD_INIT_TENSOR:       أرجع "INIT_TENSOR";
        حالة RPC_CMD_GET_ALLOC_سائيز:    أرجع "GET_ALLOC_SIZE";
        حالة RPC_CMD_HELLO:             أرجع "HELLO";
        حالة RPC_CMD_DEVICE_ڳڻپ:      أرجع "DEVICE_COUNT";
        حالة RPC_CMD_گراف_RECOMPUTE:   أرجع "GRAPH_RECOMPUTE";
        حالة RPC_CMD_MEMSET_TENSOR:     أرجع "MEMSET_TENSOR";
        حالة RPC_CMD_سيٽ_TENSOR_2_D:     أرجع "SET_TENSOR_2D";
        حالة RPC_CMD_GET_TENSOR_2D:     أرجع "GET_TENSOR_2D";
        حالة RPC_CMD_COMM_INIT:         أرجع "COMM_INIT";
        حالة RPC_CMD_COMM_ALLREDUCE:    أرجع "COMM_ALLREDUCE";
        حالة RPC_CMD_COMM_FREE:         أرجع "COMM_FREE";
        حالة RPC_CMD_NONE:              أرجع "NONE";
        افتراضي:                        أرجع "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
ثابت size_t HASH_حد = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
ثابت size_t گراف_CACHE_MAX = 1024;

بنية rpc_نياپو_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_سائيز];
};

بنية rpc_نياپو_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_سائيز];
};

بنية rpc_نياپو_device_ڳڻپ_rsp {
    uint32_t device_ڳڻپ;
};

بنية rpc_نياپو_get_alloc_سائيز_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

بنية rpc_نياپو_get_alloc_سائيز_rsp {
    uint64_t alloc_سائيز;
};

بنية rpc_نياپو_init_tensor_req {
    rpc_tensor tensor;
};

بنية rpc_نياپو_alloc_بفر_req {
    uint32_t device;
    uint64_t size;
};

بنية rpc_نياپو_alloc_بفر_rsp {
    uint64_t remote_ptr;
    uint64_t remote_سائيز;
};

بنية rpc_نياپو_get_alignment_req {
    uint32_t device;
};

بنية rpc_نياپو_get_alignment_rsp {
    uint64_t alignment;
};

بنية rpc_نياپو_get_max_سائيز_req {
    uint32_t device;
};

بنية rpc_نياپو_get_max_سائيز_rsp {
    uint64_t max_size;
};

بنية rpc_نياپو_بفر_get_بنياد_req {
    uint64_t remote_ptr;
};

بنية rpc_نياپو_بفر_get_بنياد_rsp {
    uint64_t بنياد_ptr;
};

بنية rpc_نياپو_free_بفر_req {
    uint64_t remote_ptr;
};

بنية rpc_نياپو_بفر_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

بنية rpc_نياپو_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t وٿي;
    uint64_t size;
    uint8_t value;
};

بنية rpc_نياپو_سيٽ_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t وٿي;
    uint64_t hash;
};

بنية rpc_نياپو_سيٽ_tensor_hash_rsp {
    uint8_t نتيجو;
};

بنية rpc_نياپو_get_tensor_req {
    rpc_tensor tensor;
    uint64_t وٿي;
    uint64_t size;
};

بنية rpc_نياپو_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

بنية rpc_نياپو_copy_tensor_rsp {
    uint8_t نتيجو;
};

بنية rpc_نياپو_get_device_يادگيري_req {
    uint32_t device;
};

بنية rpc_نياپو_get_device_يادگيري_rsp {
    uint64_t free_mem;
    uint64_t ڪل_mem;
};

بنية rpc_نياپو_گراف_recompute_req {
    uint32_t device;
    uint64_t uid;
};

بنية rpc_نياپو_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t وٿي;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

بنية rpc_نياپو_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t پورٽ;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    حرف     host[64];  // rank > 0: rank 0's host
};

بنية rpc_نياپو_comm_init_rsp {
    uint8_t ok;
};

بنية rpc_نياپو_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

بنية rpc_نياپو_comm_free_req {
    uint32_t device;
};

#هدايت pack(ڪڍو)

// RPC data structures

ساكن ggml_guid_t ggml_backend_rpc_guid() {
    ساكن ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    أرجع &guid;
}

بنية ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

بنية ggml_backend_rpc_بفر_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

صنف rpc_dispatcher;
بنية ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

بنية ggml_backend_rpc_بفر_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    فراغ                            * بنياد_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
ساكن uint64_t fnv_hash(ثابت uint8_t * معلومات, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    ثابت uint64_t fnv_مفرد = 0x100000001b3ULL;

    لكل (size_t i = 0; i < len; ++i) {
        hash ^= معلومات[i];
        hash *= fnv_مفرد;
    }
    أرجع hash;
}

ساكن منطقي send_نياپو(socket_ptr sock, ثابت فراغ * نياپو, size_t نياپو_سائيز) {
    إذا (!sock->send_معلومات(&نياپو_سائيز, حجم(نياپو_سائيز))) {
        أرجع خطأ;
    }
    إذا (!sock->send_معلومات(نياپو, نياپو_سائيز)) {
        أرجع خطأ;
    }
    أرجع sock->flush();
}

ساكن منطقي recv_نياپو(socket_ptr sock, فراغ * نياپو, size_t نياپو_سائيز) {
    uint64_t size;
    إذا (!sock->recv_معلومات(&size, حجم(size))) {
        أرجع خطأ;
    }
    إذا (size != نياپو_سائيز) {
        أرجع خطأ;
    }
    أرجع sock->recv_معلومات(نياپو, نياپو_سائيز);
}

ساكن منطقي recv_نياپو(socket_ptr sock, std::vector<uint8_t> & داخلا_2) {
    uint64_t size;
    إذا (!sock->recv_معلومات(&size, حجم(size))) {
        أرجع خطأ;
    }
    حاول {
        داخلا_2.resize(size);
    } امسك (ثابت std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        أرجع خطأ;
    }
    أرجع sock->recv_معلومات(داخلا_2.معلومات(), size);
}

ساكن منطقي parse_endpoint(ثابت std::string & endpoint, std::string & host, صحيح & پورٽ) {
    size_t pos = endpoint.find(':');
    إذا (pos == std::string::npos) {
        أرجع خطأ;
    }
    host = endpoint.substr(0, pos);
    حاول {
        پورٽ = std::stoi(endpoint.substr(pos + 1));
    } امسك (...) {
        أرجع خطأ;
    }
    أرجع سچ;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
ساكن منطقي send_rpc_cmd(socket_ptr sock, تعداد rpc_cmd cmd, ثابت فراغ * داخلا_2, size_t داخلا_سائيز) {
    uint8_t cmd_byte = cmd;
    إذا (!sock->send_معلومات(&cmd_byte, حجم(cmd_byte))) {
        أرجع خطأ;
    }
    إذا (!sock->send_معلومات(&داخلا_سائيز, حجم(داخلا_سائيز))) {
        أرجع خطأ;
    }
    إذا (!sock->send_معلومات(داخلا_2, داخلا_سائيز)) {
        أرجع خطأ;
    }
    أرجع sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
ساكن منطقي send_rpc_cmd(socket_ptr sock, تعداد rpc_cmd cmd, ثابت فراغ * داخلا_2, size_t داخلا_سائيز, فراغ * output, size_t output_سائيز) {
    إذا (!send_rpc_cmd(sock, cmd, داخلا_2, داخلا_سائيز)) {
        أرجع خطأ;
    }
    uint64_t ٻاهريون_سائيز;
    إذا (!sock->recv_معلومات(&ٻاهريون_سائيز, حجم(ٻاهريون_سائيز))) {
        أرجع خطأ;
    }
    إذا (ٻاهريون_سائيز != output_سائيز) {
        أرجع خطأ;
    }
    إذا (!sock->recv_معلومات(output, output_سائيز)) {
        أرجع خطأ;
    }
    أرجع سچ;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
ساكن constexpr خودڪار RPC_BUSY_SPIN_IDLE_وقت = std::chrono::milliseconds(100);

ساكن مضمن فراغ rpc_cpu_relax() {
#إذا defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ متطاير("yield" ::: "memory");
#وإلا
    std::this_thread::yield();
#جيڪڏهن_پورو
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
ساكن منطقي negotiate_hello(ثابت std::shared_ptr<socket_t> & sock) {
    rpc_نياپو_hello_req request = {};
    rpc_نياپو_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    منطقي حالت = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, حجم(request), &response, حجم(response));
    RPC_حالت_ASSERT(حالت);

    إذا (response.major != RPC_PROTO_MAJOR_نسخو || response.minor > RPC_PROTO_MINOR_نسخو) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        أرجع خطأ;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    أرجع سچ;
}

سانچو <قسم_نالو T>
صنف پيغام_قطار {
عام:
    پيغام_قطار() {}

    منطقي ڌڪو(ثابت T &value) {
        std::unique_lock<std::mutex> تالو(mutex);
        إذا (interrupted) {
            أرجع خطأ;
        }
        queue.ڌڪو(value);
        cvar.notify_all();
        أرجع سچ;
    }

    منطقي ڪڍو(T* ٻاهريون) {
        std::unique_lock<std::mutex> تالو(mutex);
        cvar.wait(تالو, [هذا] { أرجع !queue.خالي_جايون() || interrupted; });
        إذا (interrupted) {
            أرجع خطأ;
        }
        *ٻاهريون = queue.front();
        queue.ڪڍو();
        أرجع سچ;
    }

    منطقي try_ڪڍو(T* ٻاهريون) {
        std::unique_lock<std::mutex> تالو(mutex);
        إذا (interrupted || queue.خالي_جايون()) {
            أرجع خطأ;
        }
        *ٻاهريون = queue.front();
        queue.ڪڍو();
        أرجع سچ;
    }

    فراغ مداخلت() {
        std::unique_lock<std::mutex> تالو(mutex);
        interrupted = سچ;
        تالو.unlock();
        cvar.notify_all();
    }

خاص:
    منطقي interrupted = خطأ;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

صنف rpc_dispatcher {
عام:
    rpc_dispatcher() {
    }

    فراغ send(تعداد rpc_cmd cmd, std::shared_ptr<ثابت فراغ> داخلا_2, size_t داخلا_سائيز);
    فراغ send(تعداد rpc_cmd cmd, std::shared_ptr<ثابت فراغ> داخلا_2, size_t داخلا_سائيز, فراغ * output, size_t output_سائيز);
    فراغ send_async(تعداد rpc_cmd cmd, std::shared_ptr<ثابت فراغ> داخلا_2, size_t داخلا_سائيز);
    فراغ send_async(تعداد rpc_cmd cmd, std::shared_ptr<ثابت فراغ> داخلا_2, size_t داخلا_سائيز, فراغ * output, size_t output_سائيز);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    فراغ event_free(ggml_backend_event_t event);
    فراغ event_synchronize(ggml_backend_event_t event);
    فراغ event_رڪارڊ(ggml_backend_event_t event);
    فراغ synchronize();
    فراغ busy_spin_acquire();
    فراغ busy_spin_release();
    فراغ گراف_compute(uint32_t device, ثابت ggml_cgraph * cgraph);

    فراغ start(ثابت std::string & endpoint);
    فراغ work();

    ~rpc_dispatcher();

خاص:
    بنية rpc_نياپو {
        rpc_cmd                       cmd;
        std::shared_ptr<ثابت فراغ>   داخلا_2;
        size_t                        داخلا_سائيز;
        فراغ                        * output;
        size_t                        output_سائيز;
        std::promise<فراغ>            completion;
    };
    استعمال rpc_نياپو_ptr   = std::shared_ptr<rpc_نياپو>;
    استعمال rpc_نياپو_قطار = پيغام_قطار<rpc_نياپو_ptr>;
    بنية rpc_event {
        rpc_نياپو_ptr              نياپو;
        std::shared_future<فراغ> sf;
    };
    std::mutex گراف_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> گراف_uids;
    rpc_نياپو_قطار    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

ساكن فراغ rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

فراغ rpc_dispatcher::send(تعداد rpc_cmd cmd, std::shared_ptr<ثابت فراغ> داخلا_2, size_t داخلا_سائيز) {
    خودڪار نياپو = std::make_shared<rpc_نياپو>();
    نياپو->cmd = cmd;
    نياپو->داخلا_2 = داخلا_2;
    نياپو->داخلا_سائيز = داخلا_سائيز;
    نياپو->output = خالي_اشارو;
    نياپو->output_سائيز = 0;
    GGML_ASSERT(queue.ڌڪو(نياپو));
    خودڪار future = نياپو->completion.get_future();
    future.wait();
}

فراغ rpc_dispatcher::send_async(تعداد rpc_cmd cmd, std::shared_ptr<ثابت فراغ> داخلا_2, size_t داخلا_سائيز) {
    خودڪار نياپو = std::make_shared<rpc_نياپو>();
    نياپو->cmd = cmd;
    نياپو->داخلا_2 = داخلا_2;
    نياپو->داخلا_سائيز = داخلا_سائيز;
    نياپو->output = خالي_اشارو;
    نياپو->output_سائيز = 0;
    GGML_ASSERT(queue.ڌڪو(نياپو));
}

فراغ rpc_dispatcher::send(تعداد rpc_cmd cmd, std::shared_ptr<ثابت فراغ> داخلا_2, size_t داخلا_سائيز, فراغ * output, size_t output_سائيز) {
    خودڪار نياپو = std::make_shared<rpc_نياپو>();
    نياپو->cmd = cmd;
    نياپو->داخلا_2 = داخلا_2;
    نياپو->داخلا_سائيز = داخلا_سائيز;
    نياپو->output = output;
    نياپو->output_سائيز = output_سائيز;
    GGML_ASSERT(queue.ڌڪو(نياپو));
    خودڪار future = نياپو->completion.get_future();
    future.wait();
}

فراغ rpc_dispatcher::send_async(تعداد rpc_cmd cmd, std::shared_ptr<ثابت فراغ> داخلا_2, size_t داخلا_سائيز, فراغ * output, size_t output_سائيز) {
    خودڪار نياپو = std::make_shared<rpc_نياپو>();
    نياپو->cmd = cmd;
    نياپو->داخلا_2 = داخلا_2;
    نياپو->داخلا_سائيز = داخلا_سائيز;
    نياپو->output = output;
    نياپو->output_سائيز = output_سائيز;
    GGML_ASSERT(queue.ڌڪو(نياپو));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = جديد rpc_event;
    ev->نياپو = std::make_shared<rpc_نياپو>();
    ev->نياپو->cmd = RPC_CMD_NONE;
    ev->sf = ev->نياپو->completion.get_future().share();
    GGML_ASSERT(queue.ڌڪو(ev->نياپو));
    أرجع جديد ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

فراغ rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ميٽيو ev;
}

فراغ rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

فراغ rpc_dispatcher::event_رڪارڊ(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->نياپو = std::make_shared<rpc_نياپو>();
    ev->نياپو->cmd = RPC_CMD_NONE;
    ev->sf = ev->نياپو->completion.get_future().share();
    GGML_ASSERT(queue.ڌڪو(ev->نياپو));
}

فراغ rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    خودڪار نياپو = std::make_shared<rpc_نياپو>();
    نياپو->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.ڌڪو(نياپو));
    نياپو->completion.get_future().wait();
}

فراغ rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

فراغ rpc_dispatcher::busy_spin_release() {
    ثابت بنانشان اڳيون = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(اڳيون > 0);
}

فراغ rpc_dispatcher::start(ثابت std::string & endpoint) {
    std::string host;
    صحيح پورٽ;
    إذا (!parse_endpoint(endpoint, host, پورٽ)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    إذا (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), پورٽ);
    إذا (sock == خالي_اشارو) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    إذا (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = سچ;
    thread = std::thread(rpc_dispatcher_trampoline, هذا);
}

فراغ rpc_dispatcher::work() {
    خودڪار آخري_cmd = std::chrono::steady_clock::now();
    طالما (running) {
        rpc_نياپو_ptr نياپو_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        ثابت منطقي spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - آخري_cmd < RPC_BUSY_SPIN_IDLE_وقت;
        إذا (spin) {
            إذا (!queue.try_ڪڍو(&نياپو_ptr)) {
                rpc_cpu_relax();
                استمر;
            }
        } وإلا إذا (!queue.ڪڍو(&نياپو_ptr)) {
            اكسر;
        }
        إذا (نياپو_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(نياپو_ptr->cmd),
                     rpc_log_format_سائيز(نياپو_ptr->داخلا_سائيز).c_str(), rpc_log_format_سائيز(نياپو_ptr->output_سائيز).c_str());
            ثابت منطقي trace = rpc_log_debug_level() >= 3;
            ثابت خودڪار t0 = std::chrono::steady_clock::now();
            إذا (نياپو_ptr->output) {
                منطقي حالت = send_rpc_cmd(sock, نياپو_ptr->cmd, نياپو_ptr->داخلا_2.get(), نياپو_ptr->داخلا_سائيز, نياپو_ptr->output, نياپو_ptr->output_سائيز);
                RPC_حالت_ASSERT(حالت);
            } وإلا {
                منطقي حالت = send_rpc_cmd(sock, نياپو_ptr->cmd, نياپو_ptr->داخلا_2.get(), نياپو_ptr->داخلا_سائيز);
                RPC_حالت_ASSERT(حالت);
            }
            إذا (trace) {
                ثابت std::chrono::duration<مزدوج, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(نياپو_ptr->cmd), elapsed.count());
            }
        } وإلا {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        نياپو_ptr->completion.set_value();
        آخري_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = خطأ;
    queue.مداخلت();
    sock = خالي_اشارو;
    إذا (thread.joinable()) {
        thread.join();
    }
}

ساكن std::shared_ptr<rpc_dispatcher> get_dispatcher(ثابت std::string & endpoint) {
    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> تالو(mutex);
    ساكن std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    خودڪار it = dispatchers.find(endpoint);
    إذا (it != dispatchers.end()) {
        إذا (خودڪار dispatcher = it->second.تالو()) {
            أرجع dispatcher;
        }
    }

    خودڪار dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    أرجع dispatcher;
}

ساكن فراغ ggml_backend_rpc_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    خودڪار request = std::make_shared<rpc_نياپو_free_بفر_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_بفر, request, حجم(*request));
    ميٽيو ctx;
}

ساكن فراغ * ggml_backend_rpc_بفر_get_بنياد(ggml_backend_بفر_t بفر) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    إذا (ctx->بنياد_ptr != خالي_اشارو) {
        أرجع ctx->بنياد_ptr;
    }
    خودڪار request = std::make_shared<rpc_نياپو_بفر_get_بنياد_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_نياپو_بفر_get_بنياد_rsp response;
    ctx->dispatcher->send(RPC_CMD_بفر_GET_بنياد, request, حجم(*request), &response, حجم(response));
    ctx->بنياد_ptr = reinterpret_cast<فراغ *>(response.بنياد_ptr);
    أرجع ctx->بنياد_ptr;
}

ساكن منطقي ggml_backend_بفر_is_rpc(ggml_backend_بفر_t بفر) {
    أرجع بفر->iface.free_بفر == ggml_backend_rpc_بفر_free_بفر;
}

ساكن rpc_tensor serialize_tensor(ثابت ggml_tensor * tensor, ثابت rpc_dispatcher * dispatcher = خالي_اشارو) {
    rpc_tensor نتيجو;
    إذا (!tensor) {
        memset(&نتيجو, 0, حجم(نتيجو));
        أرجع نتيجو;
    }

    نتيجو.id = reinterpret_cast<uint64_t>(tensor);
    نتيجو.type = tensor->type;
    إذا (tensor->بفر && ggml_backend_بفر_is_rpc(tensor->بفر)) {
        ggml_backend_بفر_t بفر = tensor->بفر;
        ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        إذا (ctx != خالي_اشارو && (dispatcher == خالي_اشارو || ctx->dispatcher.get() == dispatcher)) {
            نتيجو.بفر = ctx->remote_ptr;
            نتيجو.معلومات = reinterpret_cast<uint64_t>(tensor->معلومات);
        } وإلا {
            نتيجو.بفر = 0;
            نتيجو.معلومات = 0;
        }
    } وإلا {
        نتيجو.بفر = 0;
        نتيجو.معلومات   = 0;
    }
    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتيجو.ne[i] = tensor->ne[i];
        نتيجو.nb[i] = tensor->nb[i];
    }
    نتيجو.op = tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتيجو.op_params[i] = tensor->op_params[i];
    }
    نتيجو.flags = tensor->flags;
    لكل (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        نتيجو.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    نتيجو.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    نتيجو.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(نتيجو.name, 0, حجم(نتيجو.name));
    نتيجو.use_ڳڻپ = 0;

    snprintf(نتيجو.name, GGML_MAX_NAME, "%s", tensor->name);
    أرجع نتيجو;
}

ساكن تعداد ggml_حالت ggml_backend_rpc_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    إذا (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == خالي_اشارو)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        خودڪار request = std::make_shared<rpc_نياپو_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, حجم(*request));
    }
    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_rpc_بفر_memset_tensor(
        ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value, size_t وٿي, size_t size) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    خودڪار request = std::make_shared<rpc_نياپو_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->وٿي = وٿي;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), وٿي,
             rpc_log_format_سائيز(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, حجم(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
ساكن std::shared_ptr<uint8_t> serialize_سيٽ_tensor(ثابت rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t وٿي, ثابت فراغ * معلومات, size_t size, size_t & داخلا_سائيز) {
    داخلا_سائيز = حجم(rpc_tensor) + حجم(cache_flag) + حجم(وٿي) + size;
    uint8_t * داخلا_2 = جديد uint8_t[داخلا_سائيز]();
    uint8_t * p = داخلا_2;
    يادگيري_نقل(p, &rpc_tensor, حجم(rpc_tensor)); p += حجم(rpc_tensor);
    يادگيري_نقل(p, &cache_flag, حجم(cache_flag)); p += حجم(cache_flag);
    يادگيري_نقل(p, &وٿي,     حجم(وٿي));     p += حجم(وٿي);
    يادگيري_نقل(p, معلومات, size);
    أرجع std::shared_ptr<uint8_t>(داخلا_2, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
ساكن منطقي rpc_use_hash_cache(ثابت ggml_tensor * tensor, size_t size) {
    أرجع size > HASH_حد && tensor->بفر->usage == GGML_BACKEND_بفر_USAGE_WEIGHTS;
}

ساكن فراغ ggml_backend_rpc_بفر_سيٽ_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), وٿي,
             rpc_log_format_سائيز(size).c_str());
    إذا (rpc_use_hash_cache(tensor, size)) {
        خودڪار request = std::make_shared<rpc_نياپو_سيٽ_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->وٿي = وٿي;
        request->hash = fnv_hash((ثابت uint8_t*)معلومات, size);
        rpc_نياپو_سيٽ_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_سيٽ_TENSOR_HASH, request, حجم(*request), &response, حجم(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.نتيجو ? "hit" : "miss");
        إذا (response.نتيجو) {
            // the server has the same data, no need to send it
            أرجع;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t داخلا_سائيز;
    خودڪار داخلا_2 = serialize_سيٽ_tensor(rpc_tensor, cache_flag, وٿي, معلومات, size, داخلا_سائيز);
    ctx->dispatcher->send(RPC_CMD_سيٽ_TENSOR, داخلا_2, داخلا_سائيز);
}

ساكن فراغ ggml_backend_rpc_بفر_سيٽ_tensor_2d(ggml_backend_بفر_t بفر, ggml_tensor * tensor, ثابت فراغ * معلومات,
        size_t وٿي, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_معلومات) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), وٿي, rpc_log_format_سائيز(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t داخلا_سائيز = حجم(rpc_tensor) + 4*حجم(uint64_t) + size*n_copies;
    uint8_t * داخلا_2 = جديد uint8_t[داخلا_سائيز]();
    uint8_t * dest = داخلا_2;
    يادگيري_نقل(dest, &rpc_tensor, حجم(rpc_tensor));
    dest += حجم(rpc_tensor);
    uint64_t سرو[4] = { وٿي, size, n_copies, stride_tensor };
    يادگيري_نقل(dest, سرو, حجم(سرو));
    dest += حجم(سرو);
    لكل (size_t i = 0; i < n_copies; i++) {
        يادگيري_نقل(dest + i*size, (ثابت حرف *)معلومات + i*stride_معلومات, size);
    }
    std::shared_ptr<uint8_t> داخلا_ptr(داخلا_2, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_سيٽ_TENSOR_2_D, داخلا_ptr, داخلا_سائيز);
}

ساكن فراغ ggml_backend_rpc_بفر_get_tensor_2d(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * معلومات,
        size_t وٿي, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_معلومات) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), وٿي, rpc_log_format_سائيز(size).c_str(), n_copies, stride_tensor);
    خودڪار request = std::make_shared<rpc_نياپو_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->وٿي   = وٿي;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    إذا (stride_معلومات == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, حجم(*request), معلومات, size*n_copies);
    } وإلا {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, حجم(*request), packed.معلومات(), packed.size());
        لكل (size_t i = 0; i < n_copies; i++) {
            يادگيري_نقل((حرف *)معلومات + i*stride_معلومات, packed.معلومات() + i*size, size);
        }
    }
}

ساكن فراغ ggml_backend_rpc_بفر_get_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * tensor, فراغ * معلومات, size_t وٿي, size_t size) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), وٿي,
             rpc_log_format_سائيز(size).c_str());
    خودڪار request = std::make_shared<rpc_نياپو_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->وٿي = وٿي;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, حجم(*request), معلومات, size);
}

ساكن منطقي ggml_backend_rpc_بفر_cpy_tensor(ggml_backend_بفر_t بفر, ثابت ggml_tensor * src, ggml_tensor * dst) {
    إذا (ggml_backend_بفر_is_rpc(src->بفر)) {
        // check if src and dst are on the same server
        ggml_backend_بفر_t src_بفر = src->بفر;
        ggml_backend_rpc_بفر_context * src_ctx = (ggml_backend_rpc_بفر_context *)src_بفر->context;
        ggml_backend_بفر_t dst_بفر = dst->بفر;
        ggml_backend_rpc_بفر_context * dst_ctx = (ggml_backend_rpc_بفر_context *)dst_بفر->context;
        إذا (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            أرجع خطأ;
        }
        ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        خودڪار request = std::make_shared<rpc_نياپو_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_نياپو_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, حجم(*request), &response, حجم(response));
        أرجع response.نتيجو;
    }
    أرجع خطأ;
}

ساكن فراغ ggml_backend_rpc_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_backend_rpc_بفر_context * ctx = (ggml_backend_rpc_بفر_context *)بفر->context;
    خودڪار request = std::make_shared<rpc_نياپو_بفر_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_بفر_CLEAR, request, حجم(*request));
}

ساكن ggml_backend_بفر_i ggml_backend_rpc_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_rpc_بفر_get_بنياد,
    /* .init_tensor     = */ ggml_backend_rpc_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_بفر_سيٽ_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_بفر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_بفر_سيٽ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_بفر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_بفر_clear,
    /* .reset           = */ عدم,
};

ساكن ثابت حرف * ggml_backend_rpc_بفر_type_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    أرجع buft_ctx->name.c_str();
}

ساكن ggml_backend_بفر_t ggml_backend_rpc_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    خودڪار request = std::make_shared<rpc_نياپو_alloc_بفر_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_نياپو_alloc_بفر_rsp response;

    خودڪار dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_بفر, request, حجم(*request), &response, حجم(response));
    إذا (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_سائيز(request->size).c_str(),
                response.remote_ptr, rpc_log_format_سائيز(response.remote_سائيز).c_str());
        ggml_backend_بفر_t بفر = ggml_backend_بفر_init(buft,
            ggml_backend_rpc_بفر_interface,
            جديد ggml_backend_rpc_بفر_context{dispatcher, خالي_اشارو, response.remote_ptr},
            response.remote_سائيز);
        أرجع بفر;
    } وإلا {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_سائيز(request->size).c_str());
        أرجع خالي_اشارو;
    }
}

ساكن size_t get_alignment(ثابت std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    خودڪار request = std::make_shared<rpc_نياپو_get_alignment_req>();
    request->device = device;
    rpc_نياپو_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, حجم(*request), &response, حجم(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_سائيز(response.alignment).c_str());
    أرجع response.alignment;
}

ساكن size_t ggml_backend_rpc_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    أرجع buft_ctx->alignment;
}

ساكن size_t get_max_سائيز(ثابت std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    خودڪار request = std::make_shared<rpc_نياپو_get_max_سائيز_req>();
    request->device = device;
    rpc_نياپو_get_max_سائيز_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_سائيز, request, حجم(*request), &response, حجم(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_سائيز(response.max_size).c_str());
    أرجع response.max_size;
}

ساكن size_t ggml_backend_rpc_get_max_سائيز(ggml_backend_بفر_type_t buft) {
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    أرجع buft_ctx->max_size;
}

ساكن size_t ggml_backend_rpc_بفر_type_get_alloc_سائيز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    منطقي rpc_get = خطأ;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == خالي_اشارو);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_سائيز_may_expand(tensor->op);

    إذا (rpc_get) {
        ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        ثابت size_t min_سائيز = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        بنية alloc_سائيز_cache_ڪنجي {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / حجم(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_سائيز_cache_ڪنجي ڪنجي = {};
        ڪنجي.device = buft_ctx->device;
        ڪنجي.type = tensor->type;
        ڪنجي.op = tensor->op;
        يادگيري_نقل(ڪنجي.op_params, tensor->op_params, حجم(ڪنجي.op_params));
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
            ڪنجي.ne[i] = (uint32_t)tensor->ne[i];
            ڪنجي.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((ثابت uint8_t *)&ڪنجي, حجم(ڪنجي));
        cache_hash = fnv_hash((ثابت uint8_t *)buft_ctx->endpoint.معلومات(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        ساكن std::mutex cache_mutex;
        ساكن std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> تالو(cache_mutex);
            خودڪار it = cache.find(cache_hash);
            إذا (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_سائيز(it->second).c_str());
                أرجع std::max<size_t>(it->second, min_سائيز);
            }
        }

        خودڪار request = std::make_shared<rpc_نياپو_get_alloc_سائيز_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_نياپو_get_alloc_سائيز_rsp response;
        خودڪار dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_سائيز, request, حجم(*request), &response, حجم(response));

        {
            std::lock_guard<std::mutex> تالو(cache_mutex);
            cache[cache_hash] = response.alloc_سائيز;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_سائيز(response.alloc_سائيز).c_str());

        أرجع std::max<size_t>(response.alloc_سائيز, min_سائيز);
    }

    أرجع ggml_nbytes(tensor);
}

ساكن ggml_backend_بفر_type_i ggml_backend_rpc_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_rpc_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_سائيز,
    /* .get_alloc_size      = */ ggml_backend_rpc_بفر_type_get_alloc_سائيز,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ساكن ثابت حرف * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    أرجع rpc_ctx->name.c_str();
}

ساكن فراغ ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ميٽيو rpc_ctx;
    ميٽيو backend;
}

ساكن فراغ ggml_backend_rpc_سيٽ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * معلومات, size_t وٿي, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), وٿي,
             rpc_log_format_سائيز(size).c_str());
    إذا (rpc_use_hash_cache(tensor, size)) {
        خودڪار request = std::make_shared<rpc_نياپو_سيٽ_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->وٿي = وٿي;
        request->hash = fnv_hash((ثابت uint8_t*)معلومات, size);
        rpc_نياپو_سيٽ_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_سيٽ_TENSOR_HASH, request, حجم(*request), &response, حجم(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.نتيجو ? "hit" : "miss");
        إذا (response.نتيجو) {
            // the server has the same data, no need to send it
            أرجع;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t داخلا_سائيز;
    خودڪار داخلا_2 = serialize_سيٽ_tensor(rpc_tensor, cache_flag, وٿي, معلومات, size, داخلا_سائيز);
    ctx->dispatcher->send_async(RPC_CMD_سيٽ_TENSOR, داخلا_2, داخلا_سائيز);
}

ساكن فراغ ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * معلومات, size_t وٿي, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), وٿي,
             rpc_log_format_سائيز(size).c_str());
    خودڪار request = std::make_shared<rpc_نياپو_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->وٿي = وٿي;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, حجم(*request), معلومات, size);
}

ساكن فراغ ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

ساكن فراغ جوڙيو_tensor(ggml_tensor * tensor, ثابت ggml_cgraph * cgraph, ثابت rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & گهميل) {
    إذا (tensor == خالي_اشارو) {
        أرجع;
    }
    إذا (گهميل.find(tensor) != گهميل.end()) {
        أرجع;
    }
    گهميل.insert(tensor);
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        جوڙيو_tensor(tensor->src[i], cgraph, dispatcher, tensors, گهميل);
    }
    جوڙيو_tensor(tensor->view_src, cgraph, dispatcher, tensors, گهميل);
    rpc_tensor نتيجو = serialize_tensor(tensor, dispatcher);
    ثابت size_t hash_pos = ggml_hash_ڳوليو(&cgraph->گهميل_hash_سيٽ, tensor);
    إذا (hash_pos != GGML_HASHSET_ڀريل && ggml_bitset_get(cgraph->گهميل_hash_سيٽ.استعمال_ٿيل, hash_pos)) {
        نتيجو.use_ڳڻپ = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(نتيجو);
}

ساكن uint8_t * serialize_گراف(uint32_t device, ثابت ggml_cgraph * cgraph, ثابت rpc_dispatcher * dispatcher, size_t * output_سائيز) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> گهميل;
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        جوڙيو_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, گهميل);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_سائيز = 2*حجم(uint32_t) + حجم(uint64_t) + n_nodes * حجم(uint64_t) + حجم(uint32_t) + n_tensors * حجم(rpc_tensor);
    uint8_t * output = جديد uint8_t[*output_سائيز]();
    uint8_t * dest = output;
    يادگيري_نقل(dest, &device, حجم(device));
    dest += حجم(device);
    يادگيري_نقل(dest, &cgraph->uid, حجم(cgraph->uid));
    dest += حجم(cgraph->uid);
    يادگيري_نقل(dest, &n_nodes, حجم(n_nodes));
    dest += حجم(n_nodes);
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        يادگيري_نقل(dest + i * حجم(uint64_t), &cgraph->nodes[i], حجم(uint64_t));
    }
    dest += n_nodes * حجم(uint64_t);
    يادگيري_نقل(dest, &n_tensors, حجم(n_tensors));
    dest += حجم(n_tensors);
    rpc_tensor * ٻاهريون_tensors = (rpc_tensor *)dest;
    يادگيري_نقل(ٻاهريون_tensors, tensors.معلومات(), n_tensors * حجم(rpc_tensor));
    أرجع output;
}

فراغ rpc_dispatcher::گراف_compute(uint32_t device, ثابت ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> تالو(گراف_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    خودڪار & device_گراف_uids = گراف_uids[device];
    منطقي reuse = cgraph->uid != 0 && device_گراف_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    إذا (reuse) {
        خودڪار request = std::make_shared<rpc_نياپو_گراف_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_گراف_RECOMPUTE, request, حجم(*request));
    } وإلا {
        إذا (cgraph->uid != 0) {
            إذا (device_گراف_uids.size() >= گراف_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_گراف_uids.size());
                device_گراف_uids.clear();
            }
            device_گراف_uids.insert(cgraph->uid);
        }
        size_t داخلا_سائيز = 0;
        uint8_t * داخلا_2 = serialize_گراف(device, cgraph, هذا, &داخلا_سائيز);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_سائيز(داخلا_سائيز).c_str());
        std::shared_ptr<uint8_t> داخلا_ptr(داخلا_2, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_گراف_COMPUTE, داخلا_ptr, داخلا_سائيز);
    }
}

ساكن تعداد ggml_حالت ggml_backend_rpc_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->گراف_compute(rpc_ctx->device, cgraph);
    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_rpc_event_رڪارڊ(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_رڪارڊ(event);
}

ساكن فراغ ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_اڻ_استعمال_ٿيل(backend);
    GGML_اڻ_استعمال_ٿيل(event);
}

ساكن فراغ ggml_backend_rpc_سيٽ_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ثابت فراغ * معلومات,
        size_t وٿي, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_معلومات) {
    ggml_backend_tensor_سيٽ_2d(tensor, معلومات, وٿي, size, n_copies, stride_tensor, stride_معلومات);
    GGML_اڻ_استعمال_ٿيل(backend);
}

ساكن فراغ ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, ثابت ggml_tensor * tensor, فراغ * معلومات,
        size_t وٿي, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_معلومات) {
    ggml_backend_tensor_get_2d(tensor, معلومات, وٿي, size, n_copies, stride_tensor, stride_معلومات);
    GGML_اڻ_استعمال_ٿيل(backend);
}

ساكن ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_سيٽ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_سيٽ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ عدم,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_rpc_گراف_compute,
    /* .event_record            = */ ggml_backend_rpc_event_رڪارڊ,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ عدم,
};

ggml_backend_بفر_type_t ggml_backend_rpc_بفر_type(ثابت حرف * endpoint, uint32_t device) {
    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> تالو(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    ساكن std::unordered_map<std::string, ggml_backend_بفر_type_t> buft_map;
    خودڪار it = buft_map.find(buft_name);
    إذا (it != buft_map.end()) {
        أرجع it->second;
    }
    خودڪار dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_سائيز(dispatcher, device);
    ggml_backend_rpc_بفر_type_context * buft_ctx = جديد ggml_backend_rpc_بفر_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    خودڪار reg = ggml_backend_rpc_جوڙيو_server(endpoint);
    ggml_backend_بفر_type_t buft = جديد ggml_backend_بفر_type {
        /* .iface   = */ ggml_backend_rpc_بفر_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_سائيز(alignment).c_str(), rpc_log_format_سائيز(max_size).c_str());
    أرجع buft;
}

ggml_backend_t ggml_backend_rpc_init(ثابت حرف * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    خودڪار dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = جديد ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    خودڪار reg = ggml_backend_rpc_جوڙيو_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = جديد ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    أرجع backend;
}

منطقي ggml_backend_is_rpc(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

فراغ ggml_backend_rpc_get_device_يادگيري(ثابت حرف * endpoint, uint32_t device, size_t * حرر, size_t * ڪل) {
    خودڪار dispatcher = get_dispatcher(endpoint);
    خودڪار request = std::make_shared<rpc_نياپو_get_device_يادگيري_req>();
    request->device = device;
    rpc_نياپو_get_device_يادگيري_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_يادگيري, request, حجم(*request), &response, حجم(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_سائيز(response.free_mem).c_str(), rpc_log_format_سائيز(response.ڪل_mem).c_str());
    *حرر = response.free_mem;
    *ڪل = response.ڪل_mem;
}

// RPC server-side implementation

صنف rpc_server {
عام:
    rpc_server(std::vector<ggml_backend_t> all_backends, ثابت حرف * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_حالتون.resize(backends.size());
    }
    ~rpc_server();

    فراغ hello(rpc_نياپو_hello_rsp & response);
    منطقي alloc_بفر(ثابت rpc_نياپو_alloc_بفر_req & request, rpc_نياپو_alloc_بفر_rsp & response);
    منطقي get_alignment(ثابت rpc_نياپو_get_alignment_req & request, rpc_نياپو_get_alignment_rsp & response);
    منطقي get_max_سائيز(ثابت rpc_نياپو_get_max_سائيز_req & request, rpc_نياپو_get_max_سائيز_rsp & response);
    منطقي بفر_get_بنياد(ثابت rpc_نياپو_بفر_get_بنياد_req & request, rpc_نياپو_بفر_get_بنياد_rsp & response);
    منطقي free_بفر(ثابت rpc_نياپو_free_بفر_req & request);
    منطقي بفر_clear(ثابت rpc_نياپو_بفر_clear_req & request);
    منطقي memset_tensor(ثابت rpc_نياپو_memset_tensor_req & request);
    منطقي سيٽ_tensor(ثابت std::vector<uint8_t> & داخلا_2);
    منطقي سيٽ_tensor_2d(ثابت std::vector<uint8_t> & داخلا_2);
    منطقي سيٽ_tensor_hash(ثابت rpc_نياپو_سيٽ_tensor_hash_req & request, rpc_نياپو_سيٽ_tensor_hash_rsp & response);
    منطقي get_tensor(ثابت rpc_نياپو_get_tensor_req & request, std::vector<uint8_t> & response);
    منطقي get_tensor_2d(ثابت rpc_نياپو_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    منطقي copy_tensor(ثابت rpc_نياپو_copy_tensor_req & request, rpc_نياپو_copy_tensor_rsp & response);
    منطقي گراف_compute(ثابت std::vector<uint8_t> & داخلا_2);
    منطقي گراف_recompute(ثابت rpc_نياپو_گراف_recompute_req & request);
    منطقي comm_init(ثابت rpc_نياپو_comm_init_req & request, rpc_نياپو_comm_init_rsp & response);
    منطقي comm_allreduce(ثابت rpc_نياپو_comm_allreduce_req & request);
    منطقي comm_free(ثابت rpc_نياپو_comm_free_req & request);
    منطقي init_tensor(ثابت rpc_نياپو_init_tensor_req & request);
    منطقي get_alloc_سائيز(ثابت rpc_نياپو_get_alloc_سائيز_req & request, rpc_نياپو_get_alloc_سائيز_rsp & response);
    منطقي get_device_يادگيري(ثابت rpc_نياپو_get_device_يادگيري_req & request, rpc_نياپو_get_device_يادگيري_rsp & response);

    بنية stored_گراف {
        std::vector<uint8_t>   بفر;
        ggml_cgraph          * graph;
    };

خاص:
    فراغ sync_all_backends();
    منطقي get_cached_file(uint64_t hash, std::vector<uint8_t> & معلومات);
    ggml_tensor * deserialize_tensor(بنية ggml_context * ctx, ثابت rpc_tensor * tensor);
    ggml_tensor * create_ڳنڍ(uint64_t id,
                              بنية ggml_context * ctx,
                              ثابت std::unordered_map<uint64_t, ثابت rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, بنية ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    بنية comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_بفر_ptr scratch;
        size_t                  scratch_سائيز = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    ثابت حرف * cache_dir;
    std::unordered_set<ggml_backend_بفر_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_گراف>> stored_graphs;
    std::vector<comm_state> comm_حالتون;
};

فراغ rpc_server::hello(rpc_نياپو_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_نسخو;
    response.minor = RPC_PROTO_MINOR_نسخو;
    response.patch = RPC_PROTO_PATCH_نسخو;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

منطقي rpc_server::get_alloc_سائيز(ثابت rpc_نياپو_get_alloc_سائيز_req & request, rpc_نياپو_get_alloc_سائيز_rsp & response) {
    uint32_t dev_id = request.device;
    إذا (dev_id >= backends.size()) {
        أرجع خطأ;
    }
    ggml_backend_بفر_type_t buft;
    بنية ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    إذا (tensor == خالي_اشارو) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        أرجع خطأ;
    }
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        إذا (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (فراغ*)tensor->بفر, tensor->معلومات);
    إذا (tensor->بفر == خالي_اشارو) {
        //No buffer allocated.
        buft = ggml_backend_get_default_بفر_type(backends[dev_id]);
    } وإلا {
        buft = tensor->بفر->buft;
    }

    response.alloc_سائيز = ggml_backend_buft_get_alloc_سائيز(buft, tensor);

    أرجع سچ;
}

منطقي rpc_server::alloc_بفر(ثابت rpc_نياپو_alloc_بفر_req & request, rpc_نياپو_alloc_بفر_rsp & response) {
    uint32_t dev_id = request.device;
    إذا (dev_id >= backends.size()) {
        أرجع خطأ;
    }
    ggml_backend_بفر_type_t buft = ggml_backend_get_default_بفر_type(backends[dev_id]);
    ggml_backend_بفر_t بفر = ggml_backend_buft_alloc_بفر(buft, request.size);
    response.remote_ptr = 0;
    response.remote_سائيز = 0;
    إذا (بفر != خالي_اشارو) {
        response.remote_ptr = reinterpret_cast<uint64_t>(بفر);
        response.remote_سائيز = بفر->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_سائيز(request.size).c_str(), response.remote_ptr,
                rpc_log_format_سائيز(response.remote_سائيز).c_str());
        buffers.insert(بفر);
    } وإلا {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_سائيز(request.size).c_str());
    }
    أرجع سچ;
}

منطقي rpc_server::get_alignment(ثابت rpc_نياپو_get_alignment_req & request, rpc_نياپو_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    إذا (dev_id >= backends.size()) {
        أرجع خطأ;
    }
    ggml_backend_بفر_type_t buft = ggml_backend_get_default_بفر_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_سائيز(alignment).c_str());
    response.alignment = alignment;
    أرجع سچ;
}

منطقي rpc_server::get_max_سائيز(ثابت rpc_نياپو_get_max_سائيز_req & request, rpc_نياپو_get_max_سائيز_rsp & response) {
    uint32_t dev_id = request.device;
    إذا (dev_id >= backends.size()) {
        أرجع خطأ;
    }
    ggml_backend_بفر_type_t buft = ggml_backend_get_default_بفر_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_سائيز(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_سائيز(max_size).c_str());
    response.max_size = max_size;
    أرجع سچ;
}

منطقي rpc_server::بفر_get_بنياد(ثابت rpc_نياپو_بفر_get_بنياد_req & request, rpc_نياپو_بفر_get_بنياد_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_بفر_t بفر = reinterpret_cast<ggml_backend_بفر_t>(request.remote_ptr);
    إذا (buffers.find(بفر) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        أرجع خطأ;
    }
    فراغ * بنياد = ggml_backend_بفر_get_بنياد(بفر);
    response.بنياد_ptr = reinterpret_cast<uint64_t>(بنياد);
    أرجع سچ;
}

منطقي rpc_server::free_بفر(ثابت rpc_نياپو_free_بفر_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_بفر_t بفر = reinterpret_cast<ggml_backend_بفر_t>(request.remote_ptr);
    إذا (buffers.find(بفر) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        أرجع خطأ;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    لكل (خودڪار & sgs : stored_graphs) {
        لكل (خودڪار & sg : sgs) {
            sg.second.graph = خالي_اشارو;
        }
    }
    ggml_backend_بفر_free(بفر);
    buffers.erase(بفر);
    أرجع سچ;
}

منطقي rpc_server::بفر_clear(ثابت rpc_نياپو_بفر_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_بفر_t بفر = reinterpret_cast<ggml_backend_بفر_t>(request.remote_ptr);
    إذا (buffers.find(بفر) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        أرجع خطأ;
    }
    ggml_backend_بفر_clear(بفر, request.value);
    أرجع سچ;
}

منطقي rpc_server::memset_tensor(ثابت rpc_نياپو_memset_tensor_req & request) {
    sync_all_backends();
    بنية ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    إذا (tensor == خالي_اشارو || tensor->بفر == خالي_اشارو) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        أرجع خطأ;
    }

    ثابت uint64_t tensor_سائيز = ggml_nbytes(tensor);
    إذا (request.وٿي > tensor_سائيز || request.size > tensor_سائيز - request.وٿي) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.وٿي, request.size, tensor_سائيز);
        أرجع خطأ;
    }

    ثابت uint64_t بفر_شروعات = (uint64_t) ggml_backend_بفر_get_بنياد(tensor->بفر);
    ثابت uint64_t بفر_سائيز = ggml_backend_بفر_get_سائيز(tensor->بفر);
    إذا (request.tensor.معلومات < بفر_شروعات) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        أرجع خطأ;
    }
    ثابت uint64_t معلومات_وٿي = request.tensor.معلومات - بفر_شروعات;
    إذا (معلومات_وٿي > بفر_سائيز ||
        request.وٿي > بفر_سائيز - معلومات_وٿي ||
        request.size > بفر_سائيز - معلومات_وٿي - request.وٿي) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        أرجع خطأ;
    }
    إذا (tensor->بفر->iface.memset_tensor == خالي_اشارو) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        أرجع خطأ;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (فراغ *) tensor->بفر, tensor->معلومات, request.وٿي,
            rpc_log_format_سائيز(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.وٿي, request.size);
    أرجع سچ;
}

ggml_tensor * rpc_server::deserialize_tensor(بنية ggml_context * ctx, ثابت rpc_tensor * tensor) {
    // Validate tensor type before using it
    إذا (tensor->type >= GGML_TYPE_ڳڻپ) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        أرجع خالي_اشارو;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    إذا (ggml_blck_سائيز((تعداد ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        أرجع خالي_اشارو;
    }

    ggml_tensor * نتيجو = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    إذا (نتيجو == خالي_اشارو) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        أرجع خالي_اشارو;
    }

    لكل (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        نتيجو->nb[i] = tensor->nb[i];
    }
    نتيجو->بفر = reinterpret_cast<ggml_backend_بفر_t>(tensor->بفر);
    إذا (نتيجو->بفر && buffers.find(نتيجو->بفر) == buffers.end()) {
        نتيجو->بفر = خالي_اشارو;
    }

    إذا (نتيجو->بفر && ggml_nelements(نتيجو) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_سائيز = (uint64_t) ggml_nbytes(نتيجو);
        uint64_t بفر_شروعات = (uint64_t) ggml_backend_بفر_get_بنياد(نتيجو->بفر);
        uint64_t بفر_سائيز = (uint64_t) ggml_backend_بفر_get_سائيز(نتيجو->بفر);
        إذا (tensor->معلومات + tensor_سائيز < tensor->معلومات ||
            tensor->معلومات < بفر_شروعات || tensor->معلومات + tensor_سائيز > بفر_شروعات + بفر_سائيز) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(نتيجو->type),
                      نتيجو->ne[0], نتيجو->ne[1], نتيجو->ne[2], نتيجو->ne[3],
                      tensor->معلومات, tensor->معلومات + tensor_سائيز, بفر_شروعات, بفر_شروعات + بفر_سائيز);
            أرجع خالي_اشارو;
        }
    }

    نتيجو->op = (ggml_op) tensor->op;
    لكل (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / حجم(int32_t); i++) {
        نتيجو->op_params[i] = tensor->op_params[i];
    }
    نتيجو->flags = tensor->flags;
    نتيجو->معلومات = reinterpret_cast<فراغ *>(tensor->معلومات);
    ggml_سيٽ_name(نتيجو, tensor->name);
    أرجع نتيجو;
}


منطقي rpc_server::سيٽ_tensor(ثابت std::vector<uint8_t> & داخلا_2) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t وٿي;
    ثابت size_t سرو_سائيز = حجم(rpc_tensor) + حجم(cache_flag) + حجم(وٿي);
    إذا (داخلا_2.size() < سرو_سائيز) {
        أرجع خطأ;
    }
    ثابت rpc_tensor * اندريون_tensor = (ثابت rpc_tensor *)داخلا_2.معلومات();
    يادگيري_نقل(&cache_flag, داخلا_2.معلومات() + حجم(rpc_tensor), حجم(cache_flag));
    يادگيري_نقل(&وٿي,     داخلا_2.معلومات() + حجم(rpc_tensor) + حجم(cache_flag), حجم(وٿي));
    ثابت size_t size = داخلا_2.size() - سرو_سائيز;

    بنية ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, اندريون_tensor);
    إذا (tensor == خالي_اشارو || tensor->بفر == خالي_اشارو) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        أرجع خطأ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (فراغ*)tensor->بفر, tensor->معلومات, وٿي,
            rpc_log_format_سائيز(size).c_str());

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_بفر_get_بنياد(tensor->بفر);
        ثابت size_t p1 = p0 + ggml_backend_بفر_get_سائيز(tensor->بفر);

        إذا (اندريون_tensor->معلومات + وٿي < p0 || اندريون_tensor->معلومات + وٿي >= p1 || size > (p1 - اندريون_tensor->معلومات - وٿي)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, اندريون_tensor->معلومات, وٿي, size, p0, p1);
            أرجع خطأ;
        }
    }

    ثابت فراغ * معلومات = داخلا_2.معلومات() + سرو_سائيز;
    إذا (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((ثابت uint8_t*)معلومات, size);
        حرف hash_str[17];
        snprintf(hash_str, حجم(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.اكتب((ثابت حرف *)معلومات, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_سيٽ(tensor, معلومات, وٿي, size);
    أرجع سچ;
}

منطقي rpc_server::سيٽ_tensor_2d(ثابت std::vector<uint8_t> & داخلا_2) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    إذا (داخلا_2.size() < حجم(rpc_tensor) + 4*حجم(uint64_t)) {
        أرجع خطأ;
    }
    ثابت rpc_tensor * اندريون_tensor = (ثابت rpc_tensor *)داخلا_2.معلومات();
    uint64_t سرو[4];
    يادگيري_نقل(سرو, داخلا_2.معلومات() + حجم(rpc_tensor), حجم(سرو));
    ثابت uint64_t وٿي   = سرو[0];
    ثابت uint64_t size     = سرو[1];
    ثابت uint64_t n_copies = سرو[2];
    ثابت uint64_t stride   = سرو[3];

    ثابت uint64_t معلومات_سائيز = داخلا_2.size() - حجم(rpc_tensor) - 4*حجم(uint64_t);
    إذا (n_copies == 0 || size == 0 || size > معلومات_سائيز / n_copies || size * n_copies != معلومات_سائيز) {
        أرجع خطأ;
    }

    بنية ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, اندريون_tensor);
    إذا (tensor == خالي_اشارو || tensor->بفر == خالي_اشارو) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        أرجع خطأ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (فراغ*)tensor->بفر, tensor->معلومات, وٿي,
            rpc_log_format_سائيز(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        إذا (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            أرجع خطأ;
        }
        ثابت uint64_t وسعت = (n_copies - 1)*stride + size;
        ثابت uint64_t p0 = (uint64_t) ggml_backend_بفر_get_بنياد(tensor->بفر);
        ثابت uint64_t p1 = p0 + ggml_backend_بفر_get_سائيز(tensor->بفر);

        إذا (اندريون_tensor->معلومات < p0 || اندريون_tensor->معلومات > p1 || وٿي > p1 - اندريون_tensor->معلومات || وسعت > p1 - اندريون_tensor->معلومات - وٿي) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, اندريون_tensor->معلومات, وٿي, وسعت, p0, p1);
            أرجع خطأ;
        }
        إذا (وٿي > ggml_nbytes(tensor) || وسعت > ggml_nbytes(tensor) - وٿي) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, وٿي, وسعت, ggml_nbytes(tensor));
            أرجع خطأ;
        }
    }

    ثابت فراغ * معلومات = داخلا_2.معلومات() + حجم(rpc_tensor) + 4*حجم(uint64_t);
    ggml_backend_tensor_سيٽ_2d(tensor, معلومات, وٿي, size, n_copies, stride, size);
    أرجع سچ;
}

منطقي rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & معلومات) {
    إذا (!cache_dir) {
        أرجع خطأ;
    }
    حرف hash_str[17];
    snprintf(hash_str, حجم(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    إذا (!fs::exists(cache_file, ec)) {
        أرجع خطأ;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    معلومات.resize(size);
    ifs.اقرأ((حرف *)معلومات.معلومات(), size);
    أرجع سچ;
}

منطقي rpc_server::سيٽ_tensor_hash(ثابت rpc_نياپو_سيٽ_tensor_hash_req & request, rpc_نياپو_سيٽ_tensor_hash_rsp & response)
{
    sync_all_backends();
    بنية ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    إذا (tensor == خالي_اشارو || tensor->بفر == خالي_اشارو) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        أرجع خطأ;
    }

    std::vector<uint8_t> cached_file;
    إذا (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.وٿي, request.hash);
        response.نتيجو = 0;
        أرجع سچ;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (فراغ*)tensor->بفر, tensor->معلومات, request.وٿي,
            rpc_log_format_سائيز(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_بفر_get_بنياد(tensor->بفر);
        ثابت size_t p1 = p0 + ggml_backend_بفر_get_سائيز(tensor->بفر);

        إذا (request.tensor.معلومات + request.وٿي < p0
         || request.tensor.معلومات + request.وٿي >= p1
         || size > (p1 - request.tensor.معلومات - request.وٿي)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.معلومات, request.وٿي, size, request.hash, p0, p1);
            أرجع خطأ;
        }
    }
    ggml_backend_tensor_سيٽ(tensor, cached_file.معلومات(), request.وٿي, size);
    response.نتيجو = 1;
    أرجع سچ;
}

منطقي rpc_server::init_tensor(ثابت rpc_نياپو_init_tensor_req & request) {
    sync_all_backends();
    بنية ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    إذا (tensor == خالي_اشارو) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        أرجع خطأ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (فراغ*)tensor->بفر, tensor->معلومات);
    // Call the backend's buffer_init_tensor function
    ggml_backend_بفر_t بفر = tensor->بفر;
    إذا (بفر && بفر->iface.init_tensor) {
        بفر->iface.init_tensor(بفر, tensor);
    } وإلا {
        إذا (!بفر) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    إذا (tensor->extra != خالي_اشارو) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        أرجع خطأ;
    }

    أرجع سچ;
}

منطقي rpc_server::get_tensor(ثابت rpc_نياپو_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    بنية ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    إذا (tensor == خالي_اشارو || tensor->بفر == خالي_اشارو) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        أرجع خطأ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (فراغ*)tensor->بفر, tensor->معلومات, request.وٿي,
             rpc_log_format_سائيز(request.size).c_str());

    // sanitize tensor->data
    {
        ثابت size_t p0 = (size_t) ggml_backend_بفر_get_بنياد(tensor->بفر);
        ثابت size_t p1 = p0 + ggml_backend_بفر_get_سائيز(tensor->بفر);

        إذا (request.tensor.معلومات + request.وٿي < p0 ||
            request.tensor.معلومات + request.وٿي >= p1 ||
            request.size > (p1 - request.tensor.معلومات - request.وٿي)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.معلومات, request.وٿي, request.size, p0, p1);
                أرجع خطأ;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.معلومات(), request.وٿي, request.size);
    أرجع سچ;
}

منطقي rpc_server::get_tensor_2d(ثابت rpc_نياپو_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    بنية ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    إذا (tensor == خالي_اشارو || tensor->بفر == خالي_اشارو) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        أرجع خطأ;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (فراغ*)tensor->بفر, tensor->معلومات, request.وٿي,
            rpc_log_format_سائيز(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        إذا (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            أرجع خطأ;
        }
        إذا (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            أرجع خطأ;
        }
        ثابت uint64_t وسعت = (request.n_copies - 1)*request.stride + request.size;
        ثابت uint64_t p0 = (uint64_t) ggml_backend_بفر_get_بنياد(tensor->بفر);
        ثابت uint64_t p1 = p0 + ggml_backend_بفر_get_سائيز(tensor->بفر);

        إذا (request.tensor.معلومات < p0 || request.tensor.معلومات > p1 || request.وٿي > p1 - request.tensor.معلومات ||
                وسعت > p1 - request.tensor.معلومات - request.وٿي) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.معلومات, request.وٿي, وسعت, p0, p1);
            أرجع خطأ;
        }
        إذا (request.وٿي > ggml_nbytes(tensor) || وسعت > ggml_nbytes(tensor) - request.وٿي) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.وٿي, وسعت, ggml_nbytes(tensor));
            أرجع خطأ;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.معلومات(), request.وٿي, request.size, request.n_copies, request.stride, request.size);
    أرجع سچ;
}

منطقي rpc_server::copy_tensor(ثابت rpc_نياپو_copy_tensor_req & request, rpc_نياپو_copy_tensor_rsp & response) {
    sync_all_backends();
    بنية ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    إذا (src == خالي_اشارو || dst == خالي_اشارو || src->بفر == خالي_اشارو || dst->بفر == خالي_اشارو) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        أرجع خطأ;
    }

    uint64_t src_سائيز   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_معلومات   = (uint64_t) dst->معلومات;
    uint64_t dst_بنياد   = (uint64_t) ggml_backend_بفر_get_بنياد(dst->بفر);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_بفر_get_سائيز(dst->بفر);

    إذا (dst_معلومات + src_سائيز > dst_بنياد + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_معلومات,
                         dst_معلومات + src_سائيز,
                         dst_بنياد,
                         dst_بنياد + dst_buf_sz);
        أرجع خطأ;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.نتيجو = ggml_backend_بفر_copy_tensor(src, dst);
    أرجع سچ;
}

ggml_tensor * rpc_server::create_ڳنڍ(uint64_t id,
                                      بنية ggml_context * ctx,
                                      ثابت std::unordered_map<uint64_t, ثابت rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, بنية ggml_tensor*> & tensor_map) {
    إذا (tensor_map.find(id) != tensor_map.end()) {
        أرجع tensor_map[id];
    }
    // Safely find the tensor pointer
    خودڪار it_ptr = tensor_ptrs.find(id);
    إذا (it_ptr == tensor_ptrs.end()) {
        أرجع خالي_اشارو;
    }
    ثابت rpc_tensor * tensor = it_ptr->second;

    بنية ggml_tensor * نتيجو = deserialize_tensor(ctx, tensor);
    إذا (نتيجو == خالي_اشارو) {
        أرجع خالي_اشارو;
    }
    إذا (نتيجو->بفر == خالي_اشارو && نتيجو->معلومات != خالي_اشارو) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        أرجع خالي_اشارو;
    }
    tensor_map[id] = نتيجو;
    لكل (صحيح i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        إذا (tensor->src[i] == 0) {
            نتيجو->src[i] = خالي_اشارو;
        } وإلا {
            نتيجو->src[i] = create_ڳنڍ(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            إذا (نتيجو->src[i] == خالي_اشارو) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                أرجع خالي_اشارو;
            }
        }
    }

    // Handle view_src similarly
    إذا (tensor->view_src == 0) {
        نتيجو->view_src = خالي_اشارو;
    } وإلا {
        نتيجو->view_src = create_ڳنڍ(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        إذا (نتيجو->view_src == خالي_اشارو) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            أرجع خالي_اشارو;
        }
    }
    نتيجو->view_offs = tensor->view_offs;
    أرجع نتيجو;
}

منطقي rpc_server::گراف_compute(ثابت std::vector<uint8_t> & داخلا_2) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    إذا (داخلا_2.size() < 2*حجم(uint32_t) + حجم(uint64_t)) {
        أرجع خطأ;
    }
    ثابت uint8_t * src = داخلا_2.معلومات();
    uint32_t device;
    يادگيري_نقل(&device, src, حجم(device));
    src += حجم(device);
    إذا (device >= backends.size()) {
        أرجع خطأ;
    }
    uint64_t uid;
    يادگيري_نقل(&uid, src, حجم(uid));
    src += حجم(uid);
    uint32_t n_nodes;
    يادگيري_نقل(&n_nodes, src, حجم(n_nodes));
    src += حجم(n_nodes);
    إذا (داخلا_2.size() < 2*حجم(uint32_t) + حجم(uint64_t) + n_nodes*حجم(uint64_t) + حجم(uint32_t)) {
        أرجع خطأ;
    }
    ثابت uint64_t * nodes = (ثابت uint64_t *)src;
    src += n_nodes*حجم(uint64_t);
    uint32_t n_tensors;
    يادگيري_نقل(&n_tensors, src, حجم(n_tensors));
    src += حجم(n_tensors);
    إذا (داخلا_2.size() < 2*حجم(uint32_t) + حجم(uint64_t) + n_nodes*حجم(uint64_t) + حجم(uint32_t) + n_tensors*حجم(rpc_tensor)) {
        أرجع خطأ;
    }
    ثابت rpc_tensor * tensors = (ثابت rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    إذا (uid != 0 && stored_graphs[device].size() >= گراف_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_گراف sg_tmp;
    stored_گراف & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_سائيز = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_گراف_overhead_custom(n_nodes, خطأ);
    إذا (sg.بفر.size() < buf_سائيز) {
        sg.بفر.resize(buf_سائيز);
    }
    بنية ggml_init_params params = {
        /*.mem_size   =*/ buf_سائيز,
        /*.mem_buffer =*/ sg.بفر.معلومات(),
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    بنية ggml_cgraph * graph = ggml_new_گراف_custom(ctx, n_nodes, خطأ);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, ثابت rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    لكل (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    ثابت خودڪار t_deserialize = std::chrono::steady_clock::now();
    لكل (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        يادگيري_نقل(&id, &nodes[i], حجم(id));
        graph->nodes[i] = create_ڳنڍ(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        إذا (graph->nodes[i] == خالي_اشارو && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            أرجع خطأ;
        }
        إذا (graph->nodes[i] != خالي_اشارو) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            ثابت size_t hash_pos = ggml_hash_وجهو(&graph->گهميل_hash_سيٽ, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_ڳڻپ;
        }
    }
    ثابت std::chrono::duration<مزدوج, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_حالت حالت = ggml_backend_گراف_compute_async(backends[device], graph);
    GGML_ASSERT(حالت == GGML_حالت_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    أرجع سچ;
}

منطقي rpc_server::گراف_recompute(ثابت rpc_نياپو_گراف_recompute_req & request) {
    uint32_t device = request.device;
    إذا (device >= backends.size()) {
        أرجع خطأ;
    }
    خودڪار it = stored_graphs[device].find(request.uid);
    إذا (it == stored_graphs[device].end() || it->second.graph == خالي_اشارو) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        أرجع خطأ;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_حالت حالت = ggml_backend_گراف_compute_async(backends[device], graph);
    GGML_ASSERT(حالت == GGML_حالت_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    أرجع سچ;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
فراغ rpc_server::sync_all_backends() {
    لكل (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
منطقي rpc_server::comm_init(ثابت rpc_نياپو_comm_init_req & request, rpc_نياپو_comm_init_rsp & response) {
    response.ok = 0;
    إذا (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        أرجع سچ;
    }
    comm_state & state = comm_حالتون[request.device];
    إذا (state.peer != خالي_اشارو) {
        response.ok = 1;
        أرجع سچ;
    }
    uint8_t local_caps[RPC_CONN_CAPS_سائيز] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_سائيز] = {};
    إذا (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.پورٽ);
        إذا (srv == خالي_اشارو) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.پورٽ);
            أرجع سچ;
        }
        state.peer = srv->accept();
        إذا (state.peer == خالي_اشارو) {
            أرجع سچ;
        }
        إذا (!state.peer->recv_معلومات(remote_caps, حجم(remote_caps))) {
            state.peer = خالي_اشارو;
            أرجع سچ;
        }
        state.peer->get_caps(local_caps);
        إذا (!state.peer->send_معلومات(local_caps, حجم(local_caps))) {
            state.peer = خالي_اشارو;
            أرجع سچ;
        }
        state.peer->update_caps(remote_caps);
    } وإلا {
        ثابت std::string host(request.host, strnlen(request.host, حجم(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        لكل (صحيح i = 0; i < 100 && state.peer == خالي_اشارو; i++) {
            state.peer = socket_t::connect(host.c_str(), request.پورٽ);
            إذا (state.peer == خالي_اشارو) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        إذا (state.peer == خالي_اشارو) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.پورٽ);
            أرجع سچ;
        }
        state.peer->get_caps(local_caps);
        إذا (!state.peer->send_معلومات(local_caps, حجم(local_caps)) ||
            !state.peer->recv_معلومات(remote_caps, حجم(remote_caps))) {
            state.peer = خالي_اشارو;
            أرجع سچ;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    أرجع سچ;
}

منطقي rpc_server::comm_allreduce(ثابت rpc_نياپو_comm_allreduce_req & request) {
    إذا (request.device >= backends.size()) {
        أرجع خطأ;
    }
    comm_state & state = comm_حالتون[request.device];
    إذا (state.peer == خالي_اشارو) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        أرجع خطأ;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_سائيز = 16*ggml_tensor_overhead() + 2*ggml_گراف_overhead_custom(8, خطأ);
    بنية ggml_init_params params = {
        /*.mem_size   =*/ ctx_سائيز,
        /*.mem_buffer =*/ عدم,
        /*.no_alloc   =*/ سچ,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != خالي_اشارو);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    إذا (t_dst == خالي_اشارو || t_dst->بفر == خالي_اشارو) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        أرجع خطأ;
    }
    ثابت size_t  nbytes = ggml_nbytes(t_dst);
    ثابت int64_t ne     = ggml_nelements(t_dst);
    إذا (nbytes == 0) {
        أرجع سچ;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    ثابت منطقي   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    ثابت size_t wire_بائيٽون = wire_bf16 ? (size_t) ne*2 : nbytes;
    ثابت size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    إذا (state.scratch_سائيز < need) {
        state.scratch.reset(ggml_backend_alloc_بفر(backend, need));
        state.scratch_سائيز = need;
    }
    حرف * scratch_بنياد = (حرف *) ggml_backend_بفر_get_بنياد(state.scratch.get());
    state.send_buf.resize(wire_بائيٽون);
    state.recv_buf.resize(wire_بائيٽون);

    خودڪار new_scratch_tensor = [&](ggml_type type, size_t وٿي) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->بفر = state.scratch.get();
        t->معلومات   = scratch_بنياد + وٿي;
        أرجع t;
    };
    خودڪار new_cpy_ڳنڍ = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->بفر = dst->بفر;
        t->معلومات   = dst->معلومات;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        أرجع t;
    };
    خودڪار compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_گراف_custom(ctx, 2, خطأ);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != خالي_اشارو ? 2 : 1;
        ggml_حالت حالت = ggml_backend_گراف_compute_async(backend, graph);
        GGML_ASSERT(حالت == GGML_حالت_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = خالي_اشارو;
    ggml_tensor * t_wire_recv = خالي_اشارو;
    إذا (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_ڳنڍ(t_dst, t_wire_send), خالي_اشارو);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.معلومات(), 0, wire_بائيٽون);
    } وإلا {
        ggml_backend_tensor_get(t_dst, state.send_buf.معلومات(), 0, wire_بائيٽون);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    إذا (state.rank == 0) {
        إذا (!state.peer->send_معلومات(state.send_buf.معلومات(), wire_بائيٽون) || !state.peer->flush() ||
            !state.peer->recv_معلومات(state.recv_buf.معلومات(), wire_بائيٽون)) {
            أرجع خطأ;
        }
    } وإلا {
        إذا (!state.peer->recv_معلومات(state.recv_buf.معلومات(), wire_بائيٽون) ||
            !state.peer->send_معلومات(state.send_buf.معلومات(), wire_بائيٽون) || !state.peer->flush()) {
            أرجع خطأ;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = خالي_اشارو;
    إذا (wire_bf16) {
        ggml_backend_tensor_سيٽ(t_wire_recv, state.recv_buf.معلومات(), 0, wire_بائيٽون);
        t_cast = new_cpy_ڳنڍ(t_wire_recv, t_peer);
    } وإلا {
        ggml_backend_tensor_سيٽ(t_peer, state.recv_buf.معلومات(), 0, wire_بائيٽون);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_جوڙيو;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->بفر = t_dst->بفر;
    t_red->معلومات   = t_dst->معلومات;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    إذا (t_cast != خالي_اشارو) {
        compute_nodes(t_cast, t_red);
    } وإلا {
        compute_nodes(t_red, خالي_اشارو);
    }
    أرجع سچ;
}

منطقي rpc_server::comm_free(ثابت rpc_نياپو_comm_free_req & request) {
    إذا (request.device >= backends.size()) {
        أرجع خطأ;
    }
    comm_حالتون[request.device] = comm_state();
    أرجع سچ;
}

منطقي rpc_server::get_device_يادگيري(ثابت rpc_نياپو_get_device_يادگيري_req & request, rpc_نياپو_get_device_يادگيري_rsp & response) {
    uint32_t dev_id = request.device;
    إذا (dev_id >= backends.size()) {
        أرجع خطأ;
    }
    size_t حرر, ڪل;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_يادگيري(dev, &حرر, &ڪل);
    response.free_mem = حرر;
    response.ڪل_mem = ڪل;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_سائيز(response.free_mem).c_str(), rpc_log_format_سائيز(response.ڪل_mem).c_str());
    أرجع سچ;
}

rpc_server::~rpc_server() {
    لكل (خودڪار بفر : buffers) {
        ggml_backend_بفر_free(بفر);
    }
}

ساكن فراغ rpc_serve_client(ثابت std::vector<ggml_backend_t> & backends, ثابت حرف * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    إذا (!sock->recv_معلومات(&cmd, 1)) {
        أرجع;
    }
    إذا (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        أرجع;
    }

    // Read input_size and validate protocol version
    uint64_t hello_داخلا_سائيز;
    إذا (!sock->recv_معلومات(&hello_داخلا_سائيز, حجم(hello_داخلا_سائيز))) {
        أرجع;
    }

    إذا (hello_داخلا_سائيز != حجم(rpc_نياپو_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_داخلا_سائيز, حجم(rpc_نياپو_hello_req), RPC_PROTO_MAJOR_نسخو);
        أرجع;
    }

    rpc_نياپو_hello_req req = {};
    إذا (!sock->recv_معلومات(&req, حجم(req))) {
        أرجع;
    }

    rpc_نياپو_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    إذا (!send_نياپو(sock, &rsp, حجم(rsp))) {
        أرجع;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    طالما (سچ) {
        إذا (!sock->recv_معلومات(&cmd, 1)) {
            اكسر;
        }
        إذا (cmd >= RPC_CMD_ڳڻپ) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            اكسر;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((تعداد rpc_cmd)cmd));
        بدل (cmd) {
            حالة RPC_CMD_HELLO: {
                // HELLO command is handled above
                أرجع;
            }
            حالة RPC_CMD_DEVICE_ڳڻپ: {
                إذا (!recv_نياپو(sock, خالي_اشارو, 0)) {
                    أرجع;
                }
                rpc_نياپو_device_ڳڻپ_rsp response;
                response.device_ڳڻپ = backends.size();
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_ALLOC_بفر: {
                rpc_نياپو_alloc_بفر_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_alloc_بفر_rsp response;
                إذا (!server.alloc_بفر(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_GET_ALLOC_سائيز: {
                rpc_نياپو_get_alloc_سائيز_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_get_alloc_سائيز_rsp response;
                إذا (!server.get_alloc_سائيز(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_GET_ALIGNMENT: {
                rpc_نياپو_get_alignment_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_get_alignment_rsp response;
                إذا (!server.get_alignment(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_GET_MAX_سائيز: {
                rpc_نياپو_get_max_سائيز_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_get_max_سائيز_rsp response;
                إذا (!server.get_max_سائيز(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_بفر_GET_بنياد: {
                rpc_نياپو_بفر_get_بنياد_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_بفر_get_بنياد_rsp response;
                إذا (!server.بفر_get_بنياد(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_FREE_بفر: {
                rpc_نياپو_free_بفر_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                إذا (!server.free_بفر(request)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_بفر_CLEAR: {
                rpc_نياپو_بفر_clear_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                إذا (!server.بفر_clear(request)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_MEMSET_TENSOR: {
                rpc_نياپو_memset_tensor_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                إذا (!server.memset_tensor(request)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_سيٽ_TENSOR: {
                std::vector<uint8_t> داخلا_2;
                إذا (!recv_نياپو(sock, داخلا_2)) {
                    أرجع;
                }
                إذا (!server.سيٽ_tensor(داخلا_2)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_سيٽ_TENSOR_2_D: {
                std::vector<uint8_t> داخلا_2;
                إذا (!recv_نياپو(sock, داخلا_2)) {
                    أرجع;
                }
                إذا (!server.سيٽ_tensor_2d(داخلا_2)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_GET_TENSOR_2D: {
                rpc_نياپو_get_tensor_2d_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                std::vector<uint8_t> response;
                إذا (!server.get_tensor_2d(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, response.معلومات(), response.size())) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_سيٽ_TENSOR_HASH: {
                rpc_نياپو_سيٽ_tensor_hash_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_سيٽ_tensor_hash_rsp response;
                إذا (!server.سيٽ_tensor_hash(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_INIT_TENSOR: {
                rpc_نياپو_init_tensor_req request;
                إذا (!recv_نياپو(sock, &request,حجم(request))) {
                    أرجع;
                }
                إذا (!server.init_tensor(request)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_GET_TENSOR: {
                rpc_نياپو_get_tensor_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                std::vector<uint8_t> response;
                إذا (!server.get_tensor(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, response.معلومات(), response.size())) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_COPY_TENSOR: {
                rpc_نياپو_copy_tensor_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_copy_tensor_rsp response;
                إذا (!server.copy_tensor(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_گراف_COMPUTE: {
                std::vector<uint8_t> داخلا_2;
                إذا (!recv_نياپو(sock, داخلا_2)) {
                    أرجع;
                }
                إذا (!server.گراف_compute(داخلا_2)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_گراف_RECOMPUTE: {
                rpc_نياپو_گراف_recompute_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                إذا (!server.گراف_recompute(request)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_COMM_INIT: {
                rpc_نياپو_comm_init_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_comm_init_rsp response;
                إذا (!server.comm_init(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_COMM_ALLREDUCE: {
                rpc_نياپو_comm_allreduce_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                إذا (!server.comm_allreduce(request)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_COMM_FREE: {
                rpc_نياپو_comm_free_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                إذا (!server.comm_free(request)) {
                    أرجع;
                }
                اكسر;
            }
            حالة RPC_CMD_GET_DEVICE_يادگيري: {
                rpc_نياپو_get_device_يادگيري_req request;
                إذا (!recv_نياپو(sock, &request, حجم(request))) {
                    أرجع;
                }
                rpc_نياپو_get_device_يادگيري_rsp response;
                إذا (!server.get_device_يادگيري(request, response)) {
                    أرجع;
                }
                إذا (!send_نياپو(sock, &response, حجم(response))) {
                    أرجع;
                }
                اكسر;
            }
            افتراضي: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((تعداد rpc_cmd)cmd));
                أرجع;
            }
        }
    }
}

فراغ ggml_backend_rpc_شروعات_server(ثابت حرف * endpoint, ثابت حرف * cache_dir,
                                   size_t n_ڌاڳا, size_t n_devices, ggml_backend_dev_t * devices) {
    إذا (n_devices == 0 || devices == خالي_اشارو) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        أرجع;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_نسخو,
        RPC_PROTO_MINOR_نسخو,
        RPC_PROTO_PATCH_نسخو);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    لكل (size_t i = 0; i < n_devices; i++) {
        خودڪار dev = devices[i];
        size_t حرر, ڪل;
        ggml_backend_dev_يادگيري(dev, &حرر, &ڪل);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 ڪل / 1024 / 1024, حرر / 1024 / 1024);
        خودڪار backend = ggml_backend_dev_init(dev, خالي_اشارو);
        إذا (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            أرجع;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : خالي_اشارو;
        إذا (reg) {
            خودڪار ggml_backend_سيٽ_n_ڌاڳا_fn = (ggml_backend_سيٽ_n_ڌاڳا_t) ggml_backend_reg_get_proc_پتو(reg, "ggml_backend_set_n_threads");
            إذا (ggml_backend_سيٽ_n_ڌاڳا_fn) {
                ggml_backend_سيٽ_n_ڌاڳا_fn(backend, n_ڌاڳا);
            }
        }
    }

    std::string host;
    صحيح پورٽ;
    إذا (!parse_endpoint(endpoint, host, پورٽ)) {
        أرجع;
    }

#جيڪڏهن_وصف_ٿيل GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#وإلا
    LOG_INFO("  transport      : TCP\n");
#جيڪڏهن_پورو // GGML_RPC_RDMA
    إذا (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        أرجع;
    }
    خودڪار server_socket = socket_t::create_server(host.c_str(), پورٽ);
    إذا (server_socket == خالي_اشارو) {
        LOG_ERROR("Failed to create server socket\n");
        أرجع;
    }
    طالما (سچ) {
        خودڪار client_socket = server_socket->accept();
        إذا (client_socket == خالي_اشارو) {
            LOG_ERROR("Failed to accept client connection\n");
            أرجع;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    لكل (خودڪار backend : backends) {
        ggml_backend_free(backend);
    }
}

ساكن ثابت حرف * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    أرجع ctx->name.c_str();
}

ساكن ثابت حرف * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    أرجع ctx->description.c_str();
}

ساكن فراغ ggml_backend_rpc_device_get_يادگيري(ggml_backend_dev_t dev, size_t * حرر, size_t * ڪل) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_يادگيري(ctx->endpoint.c_str(), ctx->device, حرر, ڪل);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_اڻ_استعمال_ٿيل(dev);
}

ساكن فراغ ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_يادگيري(dev, &props->يادگيري_free, &props->يادگيري_ڪل);
    props->caps = {
        /* .async                 = */ سچ,
        /* .host_buffer           = */ خطأ,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ سچ,
        /* .mmap_support          = */ سچ,
    };
}

ساكن ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    أرجع ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_اڻ_استعمال_ٿيل(params);
}

ساكن ggml_backend_بفر_type_t ggml_backend_rpc_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    أرجع ggml_backend_rpc_بفر_type(ctx->endpoint.c_str(), ctx->device);

    GGML_اڻ_استعمال_ٿيل(dev);
}

ساكن منطقي ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) {
    GGML_اڻ_استعمال_ٿيل(dev);
    GGML_اڻ_استعمال_ٿيل(op);
    //TODO: call the remote backend and cache the results
    أرجع سچ;
}

ساكن منطقي ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    إذا (!buft || buft->iface.get_name != ggml_backend_rpc_بفر_type_name) {
        أرجع خطأ;
    }
    ggml_backend_rpc_بفر_type_context * buft_ctx = (ggml_backend_rpc_بفر_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    أرجع buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

ساكن ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    خودڪار dispatcher = get_dispatcher(ctx->endpoint);
    أرجع dispatcher->event_new(dev);
}

ساكن فراغ ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    خودڪار dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

ساكن فراغ ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    خودڪار dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_يادگيري,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_بفر_type,
    /* .get_host_buffer_type = */ عدم,
    /* .buffer_from_host_ptr = */ عدم,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ عدم,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

بنية ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

ساكن ثابت حرف * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    أرجع ctx ? ctx->name.c_str() : "RPC";
}

ساكن size_t ggml_backend_rpc_reg_get_device_ڳڻپ(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    أرجع ctx ? ctx->devices.size() : 0;
}

ساكن ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t اشاريو) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    إذا (ctx == خالي_اشارو) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } وإلا {
        GGML_ASSERT(اشاريو < ctx->devices.size());
        أرجع ctx->devices[اشاريو];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
بنية ggml_backend_rpc_comm_context {
    بنية rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

ساكن فراغ ggml_backend_rpc_comm_free(فراغ * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    إذا (comm_ctx == خالي_اشارو) {
        أرجع;
    }
    لكل (ثابت خودڪار & rank : comm_ctx->ranks) {
        خودڪار request = std::make_shared<rpc_نياپو_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, حجم(*request));
        rank.dispatcher->busy_spin_release();
    }
    ميٽيو comm_ctx;
}

ساكن فراغ * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    إذا (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != خالي_اشارو) {
        أرجع خالي_اشارو;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    لكل (size_t i = 0; i < n_backends; i++) {
        إذا (!ggml_backend_is_rpc(backends[i])) {
            أرجع خالي_اشارو;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        لكل (ثابت خودڪار & rank : ranks) {
            إذا (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                أرجع خالي_اشارو;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    صحيح port0;
    إذا (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        أرجع خالي_اشارو;
    }
    ثابت uint32_t comm_پورٽ = (uint32_t) port0 + 1000;
    إذا (host0.size() >= 64) {
        أرجع خالي_اشارو;
    }

    لكل (ثابت خودڪار & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_نياپو_comm_init_rsp> responses(n_backends);
    لكل (size_t i = 0; i < n_backends; i++) {
        خودڪار request = std::make_shared<rpc_نياپو_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->پورٽ   = comm_پورٽ;
        إذا (i > 0) {
            يادگيري_نقل(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, حجم(*request), &responses[i], حجم(responses[i]));
    }
    لكل (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    منطقي ok = سچ;
    لكل (size_t i = 0; i < n_backends; i++) {
        إذا (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = خطأ;
        }
    }
    إذا (!ok) {
        لكل (ثابت خودڪار & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        أرجع خالي_اشارو;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    أرجع جديد ggml_backend_rpc_comm_context{std::move(ranks)};
}

ساكن منطقي ggml_backend_rpc_comm_allreduce_tensor(فراغ * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    إذا (comm_ctx == خالي_اشارو) {
        أرجع خطأ;
    }
    ثابت size_t n_ranks = comm_ctx->ranks.size();
    ثابت int64_t ne = ggml_nelements(tensors[0]);
    إذا (ne == 0) {
        أرجع سچ;
    }
    لكل (size_t i = 0; i < n_ranks; i++) {
        إذا (tensors[i] == خالي_اشارو || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->بفر == خالي_اشارو || !ggml_backend_بفر_is_rpc(tensors[i]->بفر)) {
            أرجع خطأ;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        إذا ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            أرجع خطأ;
        }
    }
    لكل (size_t i = 0; i < n_ranks; i++) {
        خودڪار request = std::make_shared<rpc_نياپو_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, حجم(*request));
    }
    أرجع سچ;
}

ساكن فراغ * ggml_backend_rpc_get_proc_پتو(ggml_backend_reg_t reg, ثابت حرف * name) {
    إذا (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        أرجع (فراغ *)ggml_backend_rpc_جوڙيو_server;
    }
    إذا (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        أرجع (فراغ *)ggml_backend_rpc_شروعات_server;
    }
    إذا (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        أرجع (فراغ *)ggml_backend_rpc_comm_init;
    }
    إذا (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        أرجع (فراغ *)ggml_backend_rpc_comm_free;
    }
    إذا (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        أرجع (فراغ *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    أرجع عدم;

    GGML_اڻ_استعمال_ٿيل(reg);
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_ڳڻپ,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_پتو,
};

ggml_backend_reg_t ggml_backend_rpc_reg(فراغ) {
    ساكن بنية ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_نسخو,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ عدم,
    };

    أرجع &ggml_backend_rpc_reg;
}

ساكن uint32_t ggml_backend_rpc_get_device_ڳڻپ(ثابت حرف * endpoint) {
    خودڪار dispatcher = get_dispatcher(endpoint);
    rpc_نياپو_device_ڳڻپ_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_ڳڻپ, خالي_اشارو, 0, &response, حجم(response));
    أرجع response.device_ڳڻپ;
}

ساكن ثابت ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_ڳڻپ,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_پتو,
};

ggml_backend_reg_t ggml_backend_rpc_جوڙيو_server(ثابت حرف * endpoint) {
    ساكن std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    ساكن std::mutex mutex;
    ساكن uint32_t dev_id = 0;
    std::lock_guard<std::mutex> تالو(mutex);
    إذا (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        أرجع reg_map[endpoint];
    }
    uint32_t dev_ڳڻپ = ggml_backend_rpc_get_device_ڳڻپ(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_ڳڻپ);
    إذا (dev_ڳڻپ == 0) {
        أرجع خالي_اشارو;
    }
    ggml_backend_rpc_reg_context * ctx = جديد ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    لكل (uint32_t ind = 0; ind < dev_ڳڻپ; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = جديد ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = جديد ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = جديد ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_نسخو,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    أرجع reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

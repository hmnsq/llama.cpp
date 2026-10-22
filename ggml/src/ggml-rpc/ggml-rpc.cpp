#అన్తర్భావ "ggml-rpc.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-cpp.h"
#అన్తర్భావ "transport.h"
#అన్తర్భావ "log.h"

#అన్తర్భావ <array>
#అన్తర్భావ <chrono>
#అన్తర్భావ <cinttypes>
#అన్తర్భావ <optional>
#అన్తర్భావ <string>
#అన్తర్భావ <thread>
#అన్తర్భావ <vector>
#అన్తర్భావ <queue>
#అన్తర్భావ <condition_variable>
#అన్తర్భావ <future>
#అన్తర్భావ <memory>
#అన్తర్భావ <mutex>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <cstring>
#అన్తర్భావ <fstream>
#అన్తర్భావ <filesystem>
#అన్తర్భావ <algorithm>
#అన్తర్భావ <atomic>

నామాకాశ fs = std::filesystem;

// macro for nicer error messages on server crash
#నిర్వచించు RPC_స్థితి_ASSERT(x) ఒకవేళ (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#సూచన pack(నెట్టు, 1)
// ggml_tensor is serialized into rpc_tensor
నిర్మాణం rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t బఫరు;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / పరిమాణం(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t దత్తాంశం;
    అక్షరం name[GGML_MAX_NAME];

    int32_t use_లెక్క;
};

static_assert(పరిమాణం(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
గణన rpc_cmd {
    RPC_CMD_ALLOC_బఫరు = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_కొలత,
    RPC_CMD_బఫరు_GET_పునాది,
    RPC_CMD_FREE_బఫరు,
    RPC_CMD_బఫరు_CLEAR,
    RPC_CMD_సమితి_TENSOR,
    RPC_CMD_సమితి_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_లేఖాచిత్రం_COMPUTE,
    RPC_CMD_GET_DEVICE_జ్ఞాపకం,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_కొలత,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_లెక్క,
    RPC_CMD_లేఖాచిత్రం_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_సమితి_TENSOR_2_D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_లెక్క,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

స్థిర స్థిరరాశి అక్షరం * rpc_cmd_name(గణన rpc_cmd cmd) {
    ఎంపిక (cmd) {
        స్థితి RPC_CMD_ALLOC_బఫరు:      తిరిగి "ALLOC_BUFFER";
        స్థితి RPC_CMD_GET_ALIGNMENT:     తిరిగి "GET_ALIGNMENT";
        స్థితి RPC_CMD_GET_MAX_కొలత:      తిరిగి "GET_MAX_SIZE";
        స్థితి RPC_CMD_బఫరు_GET_పునాది:   తిరిగి "BUFFER_GET_BASE";
        స్థితి RPC_CMD_FREE_బఫరు:       తిరిగి "FREE_BUFFER";
        స్థితి RPC_CMD_బఫరు_CLEAR:      తిరిగి "BUFFER_CLEAR";
        స్థితి RPC_CMD_సమితి_TENSOR:        తిరిగి "SET_TENSOR";
        స్థితి RPC_CMD_సమితి_TENSOR_HASH:   తిరిగి "SET_TENSOR_HASH";
        స్థితి RPC_CMD_GET_TENSOR:        తిరిగి "GET_TENSOR";
        స్థితి RPC_CMD_COPY_TENSOR:       తిరిగి "COPY_TENSOR";
        స్థితి RPC_CMD_లేఖాచిత్రం_COMPUTE:     తిరిగి "GRAPH_COMPUTE";
        స్థితి RPC_CMD_GET_DEVICE_జ్ఞాపకం: తిరిగి "GET_DEVICE_MEMORY";
        స్థితి RPC_CMD_INIT_TENSOR:       తిరిగి "INIT_TENSOR";
        స్థితి RPC_CMD_GET_ALLOC_కొలత:    తిరిగి "GET_ALLOC_SIZE";
        స్థితి RPC_CMD_HELLO:             తిరిగి "HELLO";
        స్థితి RPC_CMD_DEVICE_లెక్క:      తిరిగి "DEVICE_COUNT";
        స్థితి RPC_CMD_లేఖాచిత్రం_RECOMPUTE:   తిరిగి "GRAPH_RECOMPUTE";
        స్థితి RPC_CMD_MEMSET_TENSOR:     తిరిగి "MEMSET_TENSOR";
        స్థితి RPC_CMD_సమితి_TENSOR_2_D:     తిరిగి "SET_TENSOR_2D";
        స్థితి RPC_CMD_GET_TENSOR_2D:     తిరిగి "GET_TENSOR_2D";
        స్థితి RPC_CMD_COMM_INIT:         తిరిగి "COMM_INIT";
        స్థితి RPC_CMD_COMM_ALLREDUCE:    తిరిగి "COMM_ALLREDUCE";
        స్థితి RPC_CMD_COMM_FREE:         తిరిగి "COMM_FREE";
        స్థితి RPC_CMD_NONE:              తిరిగి "NONE";
        అప్రమేయ:                        తిరిగి "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
స్థిరరాశి size_t HASH_పరిమితి = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
స్థిరరాశి size_t లేఖాచిత్రం_CACHE_MAX = 1024;

నిర్మాణం rpc_సందేశం_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_కొలత];
};

నిర్మాణం rpc_సందేశం_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_కొలత];
};

నిర్మాణం rpc_సందేశం_device_లెక్క_rsp {
    uint32_t device_లెక్క;
};

నిర్మాణం rpc_సందేశం_get_alloc_కొలత_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

నిర్మాణం rpc_సందేశం_get_alloc_కొలత_rsp {
    uint64_t alloc_కొలత;
};

నిర్మాణం rpc_సందేశం_init_tensor_req {
    rpc_tensor tensor;
};

నిర్మాణం rpc_సందేశం_alloc_బఫరు_req {
    uint32_t device;
    uint64_t size;
};

నిర్మాణం rpc_సందేశం_alloc_బఫరు_rsp {
    uint64_t remote_ptr;
    uint64_t remote_కొలత;
};

నిర్మాణం rpc_సందేశం_get_alignment_req {
    uint32_t device;
};

నిర్మాణం rpc_సందేశం_get_alignment_rsp {
    uint64_t alignment;
};

నిర్మాణం rpc_సందేశం_get_max_కొలత_req {
    uint32_t device;
};

నిర్మాణం rpc_సందేశం_get_max_కొలత_rsp {
    uint64_t max_size;
};

నిర్మాణం rpc_సందేశం_బఫరు_get_పునాది_req {
    uint64_t remote_ptr;
};

నిర్మాణం rpc_సందేశం_బఫరు_get_పునాది_rsp {
    uint64_t పునాది_ptr;
};

నిర్మాణం rpc_సందేశం_free_బఫరు_req {
    uint64_t remote_ptr;
};

నిర్మాణం rpc_సందేశం_బఫరు_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

నిర్మాణం rpc_సందేశం_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t స్థానభ్రంశం;
    uint64_t size;
    uint8_t value;
};

నిర్మాణం rpc_సందేశం_సమితి_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t స్థానభ్రంశం;
    uint64_t hash;
};

నిర్మాణం rpc_సందేశం_సమితి_tensor_hash_rsp {
    uint8_t ఫలితం;
};

నిర్మాణం rpc_సందేశం_get_tensor_req {
    rpc_tensor tensor;
    uint64_t స్థానభ్రంశం;
    uint64_t size;
};

నిర్మాణం rpc_సందేశం_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

నిర్మాణం rpc_సందేశం_copy_tensor_rsp {
    uint8_t ఫలితం;
};

నిర్మాణం rpc_సందేశం_get_device_జ్ఞాపకం_req {
    uint32_t device;
};

నిర్మాణం rpc_సందేశం_get_device_జ్ఞాపకం_rsp {
    uint64_t free_mem;
    uint64_t మొత్తం_mem;
};

నిర్మాణం rpc_సందేశం_లేఖాచిత్రం_recompute_req {
    uint32_t device;
    uint64_t uid;
};

నిర్మాణం rpc_సందేశం_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t స్థానభ్రంశం;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

నిర్మాణం rpc_సందేశం_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t పోర్టు;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    అక్షరం     host[64];  // rank > 0: rank 0's host
};

నిర్మాణం rpc_సందేశం_comm_init_rsp {
    uint8_t ok;
};

నిర్మాణం rpc_సందేశం_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

నిర్మాణం rpc_సందేశం_comm_free_req {
    uint32_t device;
};

#సూచన pack(తీయి)

// RPC data structures

స్థిర ggml_guid_t ggml_backend_rpc_guid() {
    స్థిర ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    తిరిగి &guid;
}

నిర్మాణం ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

నిర్మాణం ggml_backend_rpc_బఫరు_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

తరగతి rpc_dispatcher;
నిర్మాణం ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

నిర్మాణం ggml_backend_rpc_బఫరు_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    శూన్య                            * పునాది_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
స్థిర uint64_t fnv_hash(స్థిరరాశి uint8_t * దత్తాంశం, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    స్థిరరాశి uint64_t fnv_ప్రధానసంఖ్య = 0x100000001b3ULL;

    కొరకు (size_t i = 0; i < len; ++i) {
        hash ^= దత్తాంశం[i];
        hash *= fnv_ప్రధానసంఖ్య;
    }
    తిరిగి hash;
}

స్థిర తార్కిక send_సందేశం(socket_ptr sock, స్థిరరాశి శూన్య * సందేశం_2, size_t సందేశం_కొలత) {
    ఒకవేళ (!sock->send_దత్తాంశం(&సందేశం_కొలత, పరిమాణం(సందేశం_కొలత))) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!sock->send_దత్తాంశం(సందేశం_2, సందేశం_కొలత)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి sock->flush();
}

స్థిర తార్కిక recv_సందేశం(socket_ptr sock, శూన్య * సందేశం_2, size_t సందేశం_కొలత) {
    uint64_t size;
    ఒకవేళ (!sock->recv_దత్తాంశం(&size, పరిమాణం(size))) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (size != సందేశం_కొలత) {
        తిరిగి అబద్ధం;
    }
    తిరిగి sock->recv_దత్తాంశం(సందేశం_2, సందేశం_కొలత);
}

స్థిర తార్కిక recv_సందేశం(socket_ptr sock, std::vector<uint8_t> & ఆదానం) {
    uint64_t size;
    ఒకవేళ (!sock->recv_దత్తాంశం(&size, పరిమాణం(size))) {
        తిరిగి అబద్ధం;
    }
    ప్రయత్నించు {
        ఆదానం.resize(size);
    } పట్టుకో (స్థిరరాశి std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        తిరిగి అబద్ధం;
    }
    తిరిగి sock->recv_దత్తాంశం(ఆదానం.దత్తాంశం(), size);
}

స్థిర తార్కిక parse_endpoint(స్థిరరాశి std::string & endpoint, std::string & host, పూర్ణాంకం & పోర్టు) {
    size_t pos = endpoint.find(':');
    ఒకవేళ (pos == std::string::npos) {
        తిరిగి అబద్ధం;
    }
    host = endpoint.substr(0, pos);
    ప్రయత్నించు {
        పోర్టు = std::stoi(endpoint.substr(pos + 1));
    } పట్టుకో (...) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
స్థిర తార్కిక send_rpc_cmd(socket_ptr sock, గణన rpc_cmd cmd, స్థిరరాశి శూన్య * ఆదానం, size_t ఆదానం_కొలత) {
    uint8_t cmd_byte = cmd;
    ఒకవేళ (!sock->send_దత్తాంశం(&cmd_byte, పరిమాణం(cmd_byte))) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!sock->send_దత్తాంశం(&ఆదానం_కొలత, పరిమాణం(ఆదానం_కొలత))) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!sock->send_దత్తాంశం(ఆదానం, ఆదానం_కొలత)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
స్థిర తార్కిక send_rpc_cmd(socket_ptr sock, గణన rpc_cmd cmd, స్థిరరాశి శూన్య * ఆదానం, size_t ఆదానం_కొలత, శూన్య * output, size_t output_కొలత) {
    ఒకవేళ (!send_rpc_cmd(sock, cmd, ఆదానం, ఆదానం_కొలత)) {
        తిరిగి అబద్ధం;
    }
    uint64_t బహిర్గతం_కొలత;
    ఒకవేళ (!sock->recv_దత్తాంశం(&బహిర్గతం_కొలత, పరిమాణం(బహిర్గతం_కొలత))) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (బహిర్గతం_కొలత != output_కొలత) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!sock->recv_దత్తాంశం(output, output_కొలత)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
స్థిర constexpr స్వయంచాలక RPC_BUSY_SPIN_IDLE_సమయం = std::chrono::milliseconds(100);

స్థిర అంతఃవరుస శూన్య rpc_cpu_relax() {
#ఒకవేళ defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ అస్థిర("yield" ::: "memory");
#లేకపోతే
    std::this_thread::yield();
#అయితే_ముగింపు
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
స్థిర తార్కిక negotiate_hello(స్థిరరాశి std::shared_ptr<socket_t> & sock) {
    rpc_సందేశం_hello_req request = {};
    rpc_సందేశం_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    తార్కిక స్థితి_2 = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, పరిమాణం(request), &response, పరిమాణం(response));
    RPC_స్థితి_ASSERT(స్థితి_2);

    ఒకవేళ (response.major != RPC_PROTO_MAJOR_సంస్కరణ || response.minor > RPC_PROTO_MINOR_సంస్కరణ) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        తిరిగి అబద్ధం;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    తిరిగి నిజం;
}

మూస <రకం_పేరు T>
తరగతి సందేశం_వరుస {
బహిరంగ:
    సందేశం_వరుస() {}

    తార్కిక నెట్టు(స్థిరరాశి T &value) {
        std::unique_lock<std::mutex> తాళం(mutex);
        ఒకవేళ (interrupted) {
            తిరిగి అబద్ధం;
        }
        queue.నెట్టు(value);
        cvar.notify_all();
        తిరిగి నిజం;
    }

    తార్కిక తీయి(T* బహిర్గతం) {
        std::unique_lock<std::mutex> తాళం(mutex);
        cvar.wait(తాళం, [ఇది] { తిరిగి !queue.ఖాళీలు() || interrupted; });
        ఒకవేళ (interrupted) {
            తిరిగి అబద్ధం;
        }
        *బహిర్గతం = queue.front();
        queue.తీయి();
        తిరిగి నిజం;
    }

    తార్కిక try_తీయి(T* బహిర్గతం) {
        std::unique_lock<std::mutex> తాళం(mutex);
        ఒకవేళ (interrupted || queue.ఖాళీలు()) {
            తిరిగి అబద్ధం;
        }
        *బహిర్గతం = queue.front();
        queue.తీయి();
        తిరిగి నిజం;
    }

    శూన్య అంతరాయం() {
        std::unique_lock<std::mutex> తాళం(mutex);
        interrupted = నిజం;
        తాళం.unlock();
        cvar.notify_all();
    }

అంతరంగ:
    తార్కిక interrupted = అబద్ధం;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

తరగతి rpc_dispatcher {
బహిరంగ:
    rpc_dispatcher() {
    }

    శూన్య send(గణన rpc_cmd cmd, std::shared_ptr<స్థిరరాశి శూన్య> ఆదానం, size_t ఆదానం_కొలత);
    శూన్య send(గణన rpc_cmd cmd, std::shared_ptr<స్థిరరాశి శూన్య> ఆదానం, size_t ఆదానం_కొలత, శూన్య * output, size_t output_కొలత);
    శూన్య send_async(గణన rpc_cmd cmd, std::shared_ptr<స్థిరరాశి శూన్య> ఆదానం, size_t ఆదానం_కొలత);
    శూన్య send_async(గణన rpc_cmd cmd, std::shared_ptr<స్థిరరాశి శూన్య> ఆదానం, size_t ఆదానం_కొలత, శూన్య * output, size_t output_కొలత);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    శూన్య event_free(ggml_backend_event_t event);
    శూన్య event_synchronize(ggml_backend_event_t event);
    శూన్య event_రికార్డు(ggml_backend_event_t event);
    శూన్య synchronize();
    శూన్య busy_spin_acquire();
    శూన్య busy_spin_release();
    శూన్య లేఖాచిత్రం_compute(uint32_t device, స్థిరరాశి ggml_cgraph * cgraph);

    శూన్య start(స్థిరరాశి std::string & endpoint);
    శూన్య work();

    ~rpc_dispatcher();

అంతరంగ:
    నిర్మాణం rpc_సందేశం {
        rpc_cmd                       cmd;
        std::shared_ptr<స్థిరరాశి శూన్య>   ఆదానం;
        size_t                        ఆదానం_కొలత;
        శూన్య                        * output;
        size_t                        output_కొలత;
        std::promise<శూన్య>            completion;
    };
    ఉపయోగించి rpc_సందేశం_ptr   = std::shared_ptr<rpc_సందేశం>;
    ఉపయోగించి rpc_సందేశం_వరుస = సందేశం_వరుస<rpc_సందేశం_ptr>;
    నిర్మాణం rpc_event {
        rpc_సందేశం_ptr              సందేశం_2;
        std::shared_future<శూన్య> sf;
    };
    std::mutex లేఖాచిత్రం_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> లేఖాచిత్రం_uids;
    rpc_సందేశం_వరుస    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

స్థిర శూన్య rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

శూన్య rpc_dispatcher::send(గణన rpc_cmd cmd, std::shared_ptr<స్థిరరాశి శూన్య> ఆదానం, size_t ఆదానం_కొలత) {
    స్వయంచాలక సందేశం_2 = std::make_shared<rpc_సందేశం>();
    సందేశం_2->cmd = cmd;
    సందేశం_2->ఆదానం = ఆదానం;
    సందేశం_2->ఆదానం_కొలత = ఆదానం_కొలత;
    సందేశం_2->output = శూన్యసూచిక;
    సందేశం_2->output_కొలత = 0;
    GGML_ASSERT(queue.నెట్టు(సందేశం_2));
    స్వయంచాలక future = సందేశం_2->completion.get_future();
    future.wait();
}

శూన్య rpc_dispatcher::send_async(గణన rpc_cmd cmd, std::shared_ptr<స్థిరరాశి శూన్య> ఆదానం, size_t ఆదానం_కొలత) {
    స్వయంచాలక సందేశం_2 = std::make_shared<rpc_సందేశం>();
    సందేశం_2->cmd = cmd;
    సందేశం_2->ఆదానం = ఆదానం;
    సందేశం_2->ఆదానం_కొలత = ఆదానం_కొలత;
    సందేశం_2->output = శూన్యసూచిక;
    సందేశం_2->output_కొలత = 0;
    GGML_ASSERT(queue.నెట్టు(సందేశం_2));
}

శూన్య rpc_dispatcher::send(గణన rpc_cmd cmd, std::shared_ptr<స్థిరరాశి శూన్య> ఆదానం, size_t ఆదానం_కొలత, శూన్య * output, size_t output_కొలత) {
    స్వయంచాలక సందేశం_2 = std::make_shared<rpc_సందేశం>();
    సందేశం_2->cmd = cmd;
    సందేశం_2->ఆదానం = ఆదానం;
    సందేశం_2->ఆదానం_కొలత = ఆదానం_కొలత;
    సందేశం_2->output = output;
    సందేశం_2->output_కొలత = output_కొలత;
    GGML_ASSERT(queue.నెట్టు(సందేశం_2));
    స్వయంచాలక future = సందేశం_2->completion.get_future();
    future.wait();
}

శూన్య rpc_dispatcher::send_async(గణన rpc_cmd cmd, std::shared_ptr<స్థిరరాశి శూన్య> ఆదానం, size_t ఆదానం_కొలత, శూన్య * output, size_t output_కొలత) {
    స్వయంచాలక సందేశం_2 = std::make_shared<rpc_సందేశం>();
    సందేశం_2->cmd = cmd;
    సందేశం_2->ఆదానం = ఆదానం;
    సందేశం_2->ఆదానం_కొలత = ఆదానం_కొలత;
    సందేశం_2->output = output;
    సందేశం_2->output_కొలత = output_కొలత;
    GGML_ASSERT(queue.నెట్టు(సందేశం_2));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = కొత్త rpc_event;
    ev->సందేశం_2 = std::make_shared<rpc_సందేశం>();
    ev->సందేశం_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->సందేశం_2->completion.get_future().share();
    GGML_ASSERT(queue.నెట్టు(ev->సందేశం_2));
    తిరిగి కొత్త ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

శూన్య rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    తొలగించు ev;
}

శూన్య rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

శూన్య rpc_dispatcher::event_రికార్డు(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->సందేశం_2 = std::make_shared<rpc_సందేశం>();
    ev->సందేశం_2->cmd = RPC_CMD_NONE;
    ev->sf = ev->సందేశం_2->completion.get_future().share();
    GGML_ASSERT(queue.నెట్టు(ev->సందేశం_2));
}

శూన్య rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    స్వయంచాలక సందేశం_2 = std::make_shared<rpc_సందేశం>();
    సందేశం_2->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.నెట్టు(సందేశం_2));
    సందేశం_2->completion.get_future().wait();
}

శూన్య rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

శూన్య rpc_dispatcher::busy_spin_release() {
    స్థిరరాశి గుర్తులేని మునుపటి = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(మునుపటి > 0);
}

శూన్య rpc_dispatcher::start(స్థిరరాశి std::string & endpoint) {
    std::string host;
    పూర్ణాంకం పోర్టు;
    ఒకవేళ (!parse_endpoint(endpoint, host, పోర్టు)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    ఒకవేళ (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), పోర్టు);
    ఒకవేళ (sock == శూన్యసూచిక) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    ఒకవేళ (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = నిజం;
    thread = std::thread(rpc_dispatcher_trampoline, ఇది);
}

శూన్య rpc_dispatcher::work() {
    స్వయంచాలక చివరి_cmd = std::chrono::steady_clock::now();
    అయితే (running) {
        rpc_సందేశం_ptr సందేశం_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        స్థిరరాశి తార్కిక spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - చివరి_cmd < RPC_BUSY_SPIN_IDLE_సమయం;
        ఒకవేళ (spin) {
            ఒకవేళ (!queue.try_తీయి(&సందేశం_ptr)) {
                rpc_cpu_relax();
                కొనసాగు;
            }
        } లేకపోతే ఒకవేళ (!queue.తీయి(&సందేశం_ptr)) {
            ఆపు;
        }
        ఒకవేళ (సందేశం_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(సందేశం_ptr->cmd),
                     rpc_log_format_కొలత(సందేశం_ptr->ఆదానం_కొలత).c_str(), rpc_log_format_కొలత(సందేశం_ptr->output_కొలత).c_str());
            స్థిరరాశి తార్కిక trace = rpc_log_debug_level() >= 3;
            స్థిరరాశి స్వయంచాలక t0 = std::chrono::steady_clock::now();
            ఒకవేళ (సందేశం_ptr->output) {
                తార్కిక స్థితి_2 = send_rpc_cmd(sock, సందేశం_ptr->cmd, సందేశం_ptr->ఆదానం.get(), సందేశం_ptr->ఆదానం_కొలత, సందేశం_ptr->output, సందేశం_ptr->output_కొలత);
                RPC_స్థితి_ASSERT(స్థితి_2);
            } లేకపోతే {
                తార్కిక స్థితి_2 = send_rpc_cmd(sock, సందేశం_ptr->cmd, సందేశం_ptr->ఆదానం.get(), సందేశం_ptr->ఆదానం_కొలత);
                RPC_స్థితి_ASSERT(స్థితి_2);
            }
            ఒకవేళ (trace) {
                స్థిరరాశి std::chrono::duration<ద్విగుణ, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(సందేశం_ptr->cmd), elapsed.count());
            }
        } లేకపోతే {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        సందేశం_ptr->completion.set_value();
        చివరి_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = అబద్ధం;
    queue.అంతరాయం();
    sock = శూన్యసూచిక;
    ఒకవేళ (thread.joinable()) {
        thread.join();
    }
}

స్థిర std::shared_ptr<rpc_dispatcher> get_dispatcher(స్థిరరాశి std::string & endpoint) {
    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);
    స్థిర std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    స్వయంచాలక it = dispatchers.find(endpoint);
    ఒకవేళ (it != dispatchers.end()) {
        ఒకవేళ (స్వయంచాలక dispatcher = it->second.తాళం()) {
            తిరిగి dispatcher;
        }
    }

    స్వయంచాలక dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    తిరిగి dispatcher;
}

స్థిర శూన్య ggml_backend_rpc_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_free_బఫరు_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_బఫరు, request, పరిమాణం(*request));
    తొలగించు ctx;
}

స్థిర శూన్య * ggml_backend_rpc_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
    ఒకవేళ (ctx->పునాది_ptr != శూన్యసూచిక) {
        తిరిగి ctx->పునాది_ptr;
    }
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_బఫరు_get_పునాది_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_సందేశం_బఫరు_get_పునాది_rsp response;
    ctx->dispatcher->send(RPC_CMD_బఫరు_GET_పునాది, request, పరిమాణం(*request), &response, పరిమాణం(response));
    ctx->పునాది_ptr = reinterpret_cast<శూన్య *>(response.పునాది_ptr);
    తిరిగి ctx->పునాది_ptr;
}

స్థిర తార్కిక ggml_backend_బఫరు_is_rpc(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి బఫరు->iface.free_బఫరు == ggml_backend_rpc_బఫరు_free_బఫరు;
}

స్థిర rpc_tensor serialize_tensor(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి rpc_dispatcher * dispatcher = శూన్యసూచిక) {
    rpc_tensor ఫలితం;
    ఒకవేళ (!tensor) {
        memset(&ఫలితం, 0, పరిమాణం(ఫలితం));
        తిరిగి ఫలితం;
    }

    ఫలితం.id = reinterpret_cast<uint64_t>(tensor);
    ఫలితం.type = tensor->type;
    ఒకవేళ (tensor->బఫరు && ggml_backend_బఫరు_is_rpc(tensor->బఫరు)) {
        ggml_backend_బఫరు_t బఫరు = tensor->బఫరు;
        ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        ఒకవేళ (ctx != శూన్యసూచిక && (dispatcher == శూన్యసూచిక || ctx->dispatcher.get() == dispatcher)) {
            ఫలితం.బఫరు = ctx->remote_ptr;
            ఫలితం.దత్తాంశం = reinterpret_cast<uint64_t>(tensor->దత్తాంశం);
        } లేకపోతే {
            ఫలితం.బఫరు = 0;
            ఫలితం.దత్తాంశం = 0;
        }
    } లేకపోతే {
        ఫలితం.బఫరు = 0;
        ఫలితం.దత్తాంశం   = 0;
    }
    కొరకు (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ఫలితం.ne[i] = tensor->ne[i];
        ఫలితం.nb[i] = tensor->nb[i];
    }
    ఫలితం.op = tensor->op;
    కొరకు (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / పరిమాణం(int32_t); i++) {
        ఫలితం.op_params[i] = tensor->op_params[i];
    }
    ఫలితం.flags = tensor->flags;
    కొరకు (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        ఫలితం.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    ఫలితం.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    ఫలితం.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(ఫలితం.name, 0, పరిమాణం(ఫలితం.name));
    ఫలితం.use_లెక్క = 0;

    snprintf(ఫలితం.name, GGML_MAX_NAME, "%s", tensor->name);
    తిరిగి ఫలితం;
}

స్థిర గణన ggml_స్థితి ggml_backend_rpc_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    ఒకవేళ (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == శూన్యసూచిక)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        స్వయంచాలక request = std::make_shared<rpc_సందేశం_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, పరిమాణం(*request));
    }
    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_rpc_బఫరు_memset_tensor(
        ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->స్థానభ్రంశం = స్థానభ్రంశం;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), స్థానభ్రంశం,
             rpc_log_format_కొలత(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, పరిమాణం(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
స్థిర std::shared_ptr<uint8_t> serialize_సమితి_tensor(స్థిరరాశి rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t స్థానభ్రంశం, స్థిరరాశి శూన్య * దత్తాంశం, size_t size, size_t & ఆదానం_కొలత) {
    ఆదానం_కొలత = పరిమాణం(rpc_tensor) + పరిమాణం(cache_flag) + పరిమాణం(స్థానభ్రంశం) + size;
    uint8_t * ఆదానం = కొత్త uint8_t[ఆదానం_కొలత]();
    uint8_t * p = ఆదానం;
    జ్ఞాపకం_నకలు(p, &rpc_tensor, పరిమాణం(rpc_tensor)); p += పరిమాణం(rpc_tensor);
    జ్ఞాపకం_నకలు(p, &cache_flag, పరిమాణం(cache_flag)); p += పరిమాణం(cache_flag);
    జ్ఞాపకం_నకలు(p, &స్థానభ్రంశం,     పరిమాణం(స్థానభ్రంశం));     p += పరిమాణం(స్థానభ్రంశం);
    జ్ఞాపకం_నకలు(p, దత్తాంశం, size);
    తిరిగి std::shared_ptr<uint8_t>(ఆదానం, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
స్థిర తార్కిక rpc_use_hash_cache(స్థిరరాశి ggml_tensor * tensor, size_t size) {
    తిరిగి size > HASH_పరిమితి && tensor->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_WEIGHTS;
}

స్థిర శూన్య ggml_backend_rpc_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), స్థానభ్రంశం,
             rpc_log_format_కొలత(size).c_str());
    ఒకవేళ (rpc_use_hash_cache(tensor, size)) {
        స్వయంచాలక request = std::make_shared<rpc_సందేశం_సమితి_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->స్థానభ్రంశం = స్థానభ్రంశం;
        request->hash = fnv_hash((స్థిరరాశి uint8_t*)దత్తాంశం, size);
        rpc_సందేశం_సమితి_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_సమితి_TENSOR_HASH, request, పరిమాణం(*request), &response, పరిమాణం(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.ఫలితం ? "hit" : "miss");
        ఒకవేళ (response.ఫలితం) {
            // the server has the same data, no need to send it
            తిరిగి;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ఆదానం_కొలత;
    స్వయంచాలక ఆదానం = serialize_సమితి_tensor(rpc_tensor, cache_flag, స్థానభ్రంశం, దత్తాంశం, size, ఆదానం_కొలత);
    ctx->dispatcher->send(RPC_CMD_సమితి_TENSOR, ఆదానం, ఆదానం_కొలత);
}

స్థిర శూన్య ggml_backend_rpc_బఫరు_సమితి_tensor_2d(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం,
        size_t స్థానభ్రంశం, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), స్థానభ్రంశం, rpc_log_format_కొలత(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t ఆదానం_కొలత = పరిమాణం(rpc_tensor) + 4*పరిమాణం(uint64_t) + size*n_copies;
    uint8_t * ఆదానం = కొత్త uint8_t[ఆదానం_కొలత]();
    uint8_t * dest = ఆదానం;
    జ్ఞాపకం_నకలు(dest, &rpc_tensor, పరిమాణం(rpc_tensor));
    dest += పరిమాణం(rpc_tensor);
    uint64_t శీర్షిక[4] = { స్థానభ్రంశం, size, n_copies, stride_tensor };
    జ్ఞాపకం_నకలు(dest, శీర్షిక, పరిమాణం(శీర్షిక));
    dest += పరిమాణం(శీర్షిక);
    కొరకు (size_t i = 0; i < n_copies; i++) {
        జ్ఞాపకం_నకలు(dest + i*size, (స్థిరరాశి అక్షరం *)దత్తాంశం + i*stride_దత్తాంశం, size);
    }
    std::shared_ptr<uint8_t> ఆదానం_ptr(ఆదానం, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_సమితి_TENSOR_2_D, ఆదానం_ptr, ఆదానం_కొలత);
}

స్థిర శూన్య ggml_backend_rpc_బఫరు_get_tensor_2d(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం,
        size_t స్థానభ్రంశం, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), స్థానభ్రంశం, rpc_log_format_కొలత(size).c_str(), n_copies, stride_tensor);
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->స్థానభ్రంశం   = స్థానభ్రంశం;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    ఒకవేళ (stride_దత్తాంశం == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, పరిమాణం(*request), దత్తాంశం, size*n_copies);
    } లేకపోతే {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, పరిమాణం(*request), packed.దత్తాంశం(), packed.size());
        కొరకు (size_t i = 0; i < n_copies; i++) {
            జ్ఞాపకం_నకలు((అక్షరం *)దత్తాంశం + i*stride_దత్తాంశం, packed.దత్తాంశం() + i*size, size);
        }
    }
}

స్థిర శూన్య ggml_backend_rpc_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), స్థానభ్రంశం,
             rpc_log_format_కొలత(size).c_str());
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->స్థానభ్రంశం = స్థానభ్రంశం;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, పరిమాణం(*request), దత్తాంశం, size);
}

స్థిర తార్కిక ggml_backend_rpc_బఫరు_cpy_tensor(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    ఒకవేళ (ggml_backend_బఫరు_is_rpc(src->బఫరు)) {
        // check if src and dst are on the same server
        ggml_backend_బఫరు_t src_బఫరు = src->బఫరు;
        ggml_backend_rpc_బఫరు_context * src_ctx = (ggml_backend_rpc_బఫరు_context *)src_బఫరు->context;
        ggml_backend_బఫరు_t dst_బఫరు = dst->బఫరు;
        ggml_backend_rpc_బఫరు_context * dst_ctx = (ggml_backend_rpc_బఫరు_context *)dst_బఫరు->context;
        ఒకవేళ (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            తిరిగి అబద్ధం;
        }
        ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        స్వయంచాలక request = std::make_shared<rpc_సందేశం_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_సందేశం_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, పరిమాణం(*request), &response, పరిమాణం(response));
        తిరిగి response.ఫలితం;
    }
    తిరిగి అబద్ధం;
}

స్థిర శూన్య ggml_backend_rpc_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    ggml_backend_rpc_బఫరు_context * ctx = (ggml_backend_rpc_బఫరు_context *)బఫరు->context;
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_బఫరు_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_బఫరు_CLEAR, request, పరిమాణం(*request));
}

స్థిర ggml_backend_బఫరు_i ggml_backend_rpc_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_rpc_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_rpc_బఫరు_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_బఫరు_సమితి_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_బఫరు_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_rpc_బఫరు_type_name(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_rpc_బఫరు_type_context * buft_ctx = (ggml_backend_rpc_బఫరు_type_context *)buft->context;
    తిరిగి buft_ctx->name.c_str();
}

స్థిర ggml_backend_బఫరు_t ggml_backend_rpc_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    ggml_backend_rpc_బఫరు_type_context * buft_ctx = (ggml_backend_rpc_బఫరు_type_context *)buft->context;
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_alloc_బఫరు_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_సందేశం_alloc_బఫరు_rsp response;

    స్వయంచాలక dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_బఫరు, request, పరిమాణం(*request), &response, పరిమాణం(response));
    ఒకవేళ (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_కొలత(request->size).c_str(),
                response.remote_ptr, rpc_log_format_కొలత(response.remote_కొలత).c_str());
        ggml_backend_బఫరు_t బఫరు = ggml_backend_బఫరు_init(buft,
            ggml_backend_rpc_బఫరు_interface,
            కొత్త ggml_backend_rpc_బఫరు_context{dispatcher, శూన్యసూచిక, response.remote_ptr},
            response.remote_కొలత);
        తిరిగి బఫరు;
    } లేకపోతే {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_కొలత(request->size).c_str());
        తిరిగి శూన్యసూచిక;
    }
}

స్థిర size_t get_alignment(స్థిరరాశి std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_get_alignment_req>();
    request->device = device;
    rpc_సందేశం_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, పరిమాణం(*request), &response, పరిమాణం(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_కొలత(response.alignment).c_str());
    తిరిగి response.alignment;
}

స్థిర size_t ggml_backend_rpc_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_rpc_బఫరు_type_context * buft_ctx = (ggml_backend_rpc_బఫరు_type_context *)buft->context;
    తిరిగి buft_ctx->alignment;
}

స్థిర size_t get_max_కొలత(స్థిరరాశి std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_get_max_కొలత_req>();
    request->device = device;
    rpc_సందేశం_get_max_కొలత_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_కొలత, request, పరిమాణం(*request), &response, పరిమాణం(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_కొలత(response.max_size).c_str());
    తిరిగి response.max_size;
}

స్థిర size_t ggml_backend_rpc_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_rpc_బఫరు_type_context * buft_ctx = (ggml_backend_rpc_బఫరు_type_context *)buft->context;
    తిరిగి buft_ctx->max_size;
}

స్థిర size_t ggml_backend_rpc_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    తార్కిక rpc_get = అబద్ధం;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == శూన్యసూచిక);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_కొలత_may_expand(tensor->op);

    ఒకవేళ (rpc_get) {
        ggml_backend_rpc_బఫరు_type_context * buft_ctx = (ggml_backend_rpc_బఫరు_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        స్థిరరాశి size_t min_కొలత = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        నిర్మాణం alloc_కొలత_cache_కీలకం {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / పరిమాణం(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_కొలత_cache_కీలకం కీలకం = {};
        కీలకం.device = buft_ctx->device;
        కీలకం.type = tensor->type;
        కీలకం.op = tensor->op;
        జ్ఞాపకం_నకలు(కీలకం.op_params, tensor->op_params, పరిమాణం(కీలకం.op_params));
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
            కీలకం.ne[i] = (uint32_t)tensor->ne[i];
            కీలకం.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((స్థిరరాశి uint8_t *)&కీలకం, పరిమాణం(కీలకం));
        cache_hash = fnv_hash((స్థిరరాశి uint8_t *)buft_ctx->endpoint.దత్తాంశం(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        స్థిర std::mutex cache_mutex;
        స్థిర std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> తాళం(cache_mutex);
            స్వయంచాలక it = cache.find(cache_hash);
            ఒకవేళ (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_కొలత(it->second).c_str());
                తిరిగి std::max<size_t>(it->second, min_కొలత);
            }
        }

        స్వయంచాలక request = std::make_shared<rpc_సందేశం_get_alloc_కొలత_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_సందేశం_get_alloc_కొలత_rsp response;
        స్వయంచాలక dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_కొలత, request, పరిమాణం(*request), &response, పరిమాణం(response));

        {
            std::lock_guard<std::mutex> తాళం(cache_mutex);
            cache[cache_hash] = response.alloc_కొలత;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_కొలత(response.alloc_కొలత).c_str());

        తిరిగి std::max<size_t>(response.alloc_కొలత, min_కొలత);
    }

    తిరిగి ggml_nbytes(tensor);
}

స్థిర ggml_backend_బఫరు_type_i ggml_backend_rpc_బఫరు_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_బఫరు_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ఖాళీ,
    /* .get_alignment       = */ ggml_backend_rpc_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_rpc_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ఖాళీ,
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    తిరిగి rpc_ctx->name.c_str();
}

స్థిర శూన్య ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    తొలగించు rpc_ctx;
    తొలగించు backend;
}

స్థిర శూన్య ggml_backend_rpc_సమితి_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), స్థానభ్రంశం,
             rpc_log_format_కొలత(size).c_str());
    ఒకవేళ (rpc_use_hash_cache(tensor, size)) {
        స్వయంచాలక request = std::make_shared<rpc_సందేశం_సమితి_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->స్థానభ్రంశం = స్థానభ్రంశం;
        request->hash = fnv_hash((స్థిరరాశి uint8_t*)దత్తాంశం, size);
        rpc_సందేశం_సమితి_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_సమితి_TENSOR_HASH, request, పరిమాణం(*request), &response, పరిమాణం(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.ఫలితం ? "hit" : "miss");
        ఒకవేళ (response.ఫలితం) {
            // the server has the same data, no need to send it
            తిరిగి;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t ఆదానం_కొలత;
    స్వయంచాలక ఆదానం = serialize_సమితి_tensor(rpc_tensor, cache_flag, స్థానభ్రంశం, దత్తాంశం, size, ఆదానం_కొలత);
    ctx->dispatcher->send_async(RPC_CMD_సమితి_TENSOR, ఆదానం, ఆదానం_కొలత);
}

స్థిర శూన్య ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), స్థానభ్రంశం,
             rpc_log_format_కొలత(size).c_str());
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->స్థానభ్రంశం = స్థానభ్రంశం;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, పరిమాణం(*request), దత్తాంశం, size);
}

స్థిర శూన్య ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

స్థిర శూన్య కూడు_tensor(ggml_tensor * tensor, స్థిరరాశి ggml_cgraph * cgraph, స్థిరరాశి rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & సందర్శించిన) {
    ఒకవేళ (tensor == శూన్యసూచిక) {
        తిరిగి;
    }
    ఒకవేళ (సందర్శించిన.find(tensor) != సందర్శించిన.end()) {
        తిరిగి;
    }
    సందర్శించిన.insert(tensor);
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        కూడు_tensor(tensor->src[i], cgraph, dispatcher, tensors, సందర్శించిన);
    }
    కూడు_tensor(tensor->view_src, cgraph, dispatcher, tensors, సందర్శించిన);
    rpc_tensor ఫలితం = serialize_tensor(tensor, dispatcher);
    స్థిరరాశి size_t hash_pos = ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, tensor);
    ఒకవేళ (hash_pos != GGML_HASHSET_నిండిన && ggml_bitset_get(cgraph->సందర్శించిన_hash_సమితి.వాడిన, hash_pos)) {
        ఫలితం.use_లెక్క = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(ఫలితం);
}

స్థిర uint8_t * serialize_లేఖాచిత్రం(uint32_t device, స్థిరరాశి ggml_cgraph * cgraph, స్థిరరాశి rpc_dispatcher * dispatcher, size_t * output_కొలత) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> సందర్శించిన;
    కొరకు (uint32_t i = 0; i < n_nodes; i++) {
        కూడు_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, సందర్శించిన);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_కొలత = 2*పరిమాణం(uint32_t) + పరిమాణం(uint64_t) + n_nodes * పరిమాణం(uint64_t) + పరిమాణం(uint32_t) + n_tensors * పరిమాణం(rpc_tensor);
    uint8_t * output = కొత్త uint8_t[*output_కొలత]();
    uint8_t * dest = output;
    జ్ఞాపకం_నకలు(dest, &device, పరిమాణం(device));
    dest += పరిమాణం(device);
    జ్ఞాపకం_నకలు(dest, &cgraph->uid, పరిమాణం(cgraph->uid));
    dest += పరిమాణం(cgraph->uid);
    జ్ఞాపకం_నకలు(dest, &n_nodes, పరిమాణం(n_nodes));
    dest += పరిమాణం(n_nodes);
    కొరకు (uint32_t i = 0; i < n_nodes; i++) {
        జ్ఞాపకం_నకలు(dest + i * పరిమాణం(uint64_t), &cgraph->nodes[i], పరిమాణం(uint64_t));
    }
    dest += n_nodes * పరిమాణం(uint64_t);
    జ్ఞాపకం_నకలు(dest, &n_tensors, పరిమాణం(n_tensors));
    dest += పరిమాణం(n_tensors);
    rpc_tensor * బహిర్గతం_tensors = (rpc_tensor *)dest;
    జ్ఞాపకం_నకలు(బహిర్గతం_tensors, tensors.దత్తాంశం(), n_tensors * పరిమాణం(rpc_tensor));
    తిరిగి output;
}

శూన్య rpc_dispatcher::లేఖాచిత్రం_compute(uint32_t device, స్థిరరాశి ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> తాళం(లేఖాచిత్రం_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    స్వయంచాలక & device_లేఖాచిత్రం_uids = లేఖాచిత్రం_uids[device];
    తార్కిక reuse = cgraph->uid != 0 && device_లేఖాచిత్రం_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    ఒకవేళ (reuse) {
        స్వయంచాలక request = std::make_shared<rpc_సందేశం_లేఖాచిత్రం_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_లేఖాచిత్రం_RECOMPUTE, request, పరిమాణం(*request));
    } లేకపోతే {
        ఒకవేళ (cgraph->uid != 0) {
            ఒకవేళ (device_లేఖాచిత్రం_uids.size() >= లేఖాచిత్రం_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_లేఖాచిత్రం_uids.size());
                device_లేఖాచిత్రం_uids.clear();
            }
            device_లేఖాచిత్రం_uids.insert(cgraph->uid);
        }
        size_t ఆదానం_కొలత = 0;
        uint8_t * ఆదానం = serialize_లేఖాచిత్రం(device, cgraph, ఇది, &ఆదానం_కొలత);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_కొలత(ఆదానం_కొలత).c_str());
        std::shared_ptr<uint8_t> ఆదానం_ptr(ఆదానం, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_లేఖాచిత్రం_COMPUTE, ఆదానం_ptr, ఆదానం_కొలత);
    }
}

స్థిర గణన ggml_స్థితి ggml_backend_rpc_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->లేఖాచిత్రం_compute(rpc_ctx->device, cgraph);
    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_rpc_event_రికార్డు(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_రికార్డు(event);
}

స్థిర శూన్య ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_వాడని(backend);
    GGML_వాడని(event);
}

స్థిర శూన్య ggml_backend_rpc_సమితి_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం,
        size_t స్థానభ్రంశం, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    ggml_backend_tensor_సమితి_2d(tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం);
    GGML_వాడని(backend);
}

స్థిర శూన్య ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, స్థిరరాశి ggml_tensor * tensor, శూన్య * దత్తాంశం,
        size_t స్థానభ్రంశం, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం) {
    ggml_backend_tensor_get_2d(tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం);
    GGML_వాడని(backend);
}

స్థిర ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_సమితి_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_సమితి_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ఖాళీ,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_rpc_లేఖాచిత్రం_compute,
    /* .event_record            = */ ggml_backend_rpc_event_రికార్డు,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ ఖాళీ,
};

ggml_backend_బఫరు_type_t ggml_backend_rpc_బఫరు_type(స్థిరరాశి అక్షరం * endpoint, uint32_t device) {
    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    స్థిర std::unordered_map<std::string, ggml_backend_బఫరు_type_t> buft_map;
    స్వయంచాలక it = buft_map.find(buft_name);
    ఒకవేళ (it != buft_map.end()) {
        తిరిగి it->second;
    }
    స్వయంచాలక dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_కొలత(dispatcher, device);
    ggml_backend_rpc_బఫరు_type_context * buft_ctx = కొత్త ggml_backend_rpc_బఫరు_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    స్వయంచాలక reg = ggml_backend_rpc_కూడు_server(endpoint);
    ggml_backend_బఫరు_type_t buft = కొత్త ggml_backend_బఫరు_type {
        /* .iface   = */ ggml_backend_rpc_బఫరు_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_కొలత(alignment).c_str(), rpc_log_format_కొలత(max_size).c_str());
    తిరిగి buft;
}

ggml_backend_t ggml_backend_rpc_init(స్థిరరాశి అక్షరం * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    స్వయంచాలక dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = కొత్త ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    స్వయంచాలక reg = ggml_backend_rpc_కూడు_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = కొత్త ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    తిరిగి backend;
}

తార్కిక ggml_backend_is_rpc(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

శూన్య ggml_backend_rpc_get_device_జ్ఞాపకం(స్థిరరాశి అక్షరం * endpoint, uint32_t device, size_t * విడుదల, size_t * మొత్తం) {
    స్వయంచాలక dispatcher = get_dispatcher(endpoint);
    స్వయంచాలక request = std::make_shared<rpc_సందేశం_get_device_జ్ఞాపకం_req>();
    request->device = device;
    rpc_సందేశం_get_device_జ్ఞాపకం_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_జ్ఞాపకం, request, పరిమాణం(*request), &response, పరిమాణం(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_కొలత(response.free_mem).c_str(), rpc_log_format_కొలత(response.మొత్తం_mem).c_str());
    *విడుదల = response.free_mem;
    *మొత్తం = response.మొత్తం_mem;
}

// RPC server-side implementation

తరగతి rpc_server {
బహిరంగ:
    rpc_server(std::vector<ggml_backend_t> all_backends, స్థిరరాశి అక్షరం * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_స్థితులు.resize(backends.size());
    }
    ~rpc_server();

    శూన్య hello(rpc_సందేశం_hello_rsp & response);
    తార్కిక alloc_బఫరు(స్థిరరాశి rpc_సందేశం_alloc_బఫరు_req & request, rpc_సందేశం_alloc_బఫరు_rsp & response);
    తార్కిక get_alignment(స్థిరరాశి rpc_సందేశం_get_alignment_req & request, rpc_సందేశం_get_alignment_rsp & response);
    తార్కిక get_max_కొలత(స్థిరరాశి rpc_సందేశం_get_max_కొలత_req & request, rpc_సందేశం_get_max_కొలత_rsp & response);
    తార్కిక బఫరు_get_పునాది(స్థిరరాశి rpc_సందేశం_బఫరు_get_పునాది_req & request, rpc_సందేశం_బఫరు_get_పునాది_rsp & response);
    తార్కిక free_బఫరు(స్థిరరాశి rpc_సందేశం_free_బఫరు_req & request);
    తార్కిక బఫరు_clear(స్థిరరాశి rpc_సందేశం_బఫరు_clear_req & request);
    తార్కిక memset_tensor(స్థిరరాశి rpc_సందేశం_memset_tensor_req & request);
    తార్కిక సమితి_tensor(స్థిరరాశి std::vector<uint8_t> & ఆదానం);
    తార్కిక సమితి_tensor_2d(స్థిరరాశి std::vector<uint8_t> & ఆదానం);
    తార్కిక సమితి_tensor_hash(స్థిరరాశి rpc_సందేశం_సమితి_tensor_hash_req & request, rpc_సందేశం_సమితి_tensor_hash_rsp & response);
    తార్కిక get_tensor(స్థిరరాశి rpc_సందేశం_get_tensor_req & request, std::vector<uint8_t> & response);
    తార్కిక get_tensor_2d(స్థిరరాశి rpc_సందేశం_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    తార్కిక copy_tensor(స్థిరరాశి rpc_సందేశం_copy_tensor_req & request, rpc_సందేశం_copy_tensor_rsp & response);
    తార్కిక లేఖాచిత్రం_compute(స్థిరరాశి std::vector<uint8_t> & ఆదానం);
    తార్కిక లేఖాచిత్రం_recompute(స్థిరరాశి rpc_సందేశం_లేఖాచిత్రం_recompute_req & request);
    తార్కిక comm_init(స్థిరరాశి rpc_సందేశం_comm_init_req & request, rpc_సందేశం_comm_init_rsp & response);
    తార్కిక comm_allreduce(స్థిరరాశి rpc_సందేశం_comm_allreduce_req & request);
    తార్కిక comm_free(స్థిరరాశి rpc_సందేశం_comm_free_req & request);
    తార్కిక init_tensor(స్థిరరాశి rpc_సందేశం_init_tensor_req & request);
    తార్కిక get_alloc_కొలత(స్థిరరాశి rpc_సందేశం_get_alloc_కొలత_req & request, rpc_సందేశం_get_alloc_కొలత_rsp & response);
    తార్కిక get_device_జ్ఞాపకం(స్థిరరాశి rpc_సందేశం_get_device_జ్ఞాపకం_req & request, rpc_సందేశం_get_device_జ్ఞాపకం_rsp & response);

    నిర్మాణం stored_లేఖాచిత్రం {
        std::vector<uint8_t>   బఫరు;
        ggml_cgraph          * graph;
    };

అంతరంగ:
    శూన్య sync_all_backends();
    తార్కిక get_cached_file(uint64_t hash, std::vector<uint8_t> & దత్తాంశం);
    ggml_tensor * deserialize_tensor(నిర్మాణం ggml_context * ctx, స్థిరరాశి rpc_tensor * tensor);
    ggml_tensor * create_కణుపు(uint64_t id,
                              నిర్మాణం ggml_context * ctx,
                              స్థిరరాశి std::unordered_map<uint64_t, స్థిరరాశి rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, నిర్మాణం ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    నిర్మాణం comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_బఫరు_ptr scratch;
        size_t                  scratch_కొలత = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    స్థిరరాశి అక్షరం * cache_dir;
    std::unordered_set<ggml_backend_బఫరు_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_లేఖాచిత్రం>> stored_graphs;
    std::vector<comm_state> comm_స్థితులు;
};

శూన్య rpc_server::hello(rpc_సందేశం_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_సంస్కరణ;
    response.minor = RPC_PROTO_MINOR_సంస్కరణ;
    response.patch = RPC_PROTO_PATCH_సంస్కరణ;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

తార్కిక rpc_server::get_alloc_కొలత(స్థిరరాశి rpc_సందేశం_get_alloc_కొలత_req & request, rpc_సందేశం_get_alloc_కొలత_rsp & response) {
    uint32_t dev_id = request.device;
    ఒకవేళ (dev_id >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    ggml_backend_బఫరు_type_t buft;
    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ఒకవేళ (tensor == శూన్యసూచిక) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        తిరిగి అబద్ధం;
    }
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఒకవేళ (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (శూన్య*)tensor->బఫరు, tensor->దత్తాంశం);
    ఒకవేళ (tensor->బఫరు == శూన్యసూచిక) {
        //No buffer allocated.
        buft = ggml_backend_get_default_బఫరు_type(backends[dev_id]);
    } లేకపోతే {
        buft = tensor->బఫరు->buft;
    }

    response.alloc_కొలత = ggml_backend_buft_get_alloc_కొలత(buft, tensor);

    తిరిగి నిజం;
}

తార్కిక rpc_server::alloc_బఫరు(స్థిరరాశి rpc_సందేశం_alloc_బఫరు_req & request, rpc_సందేశం_alloc_బఫరు_rsp & response) {
    uint32_t dev_id = request.device;
    ఒకవేళ (dev_id >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    ggml_backend_బఫరు_type_t buft = ggml_backend_get_default_బఫరు_type(backends[dev_id]);
    ggml_backend_బఫరు_t బఫరు = ggml_backend_buft_alloc_బఫరు(buft, request.size);
    response.remote_ptr = 0;
    response.remote_కొలత = 0;
    ఒకవేళ (బఫరు != శూన్యసూచిక) {
        response.remote_ptr = reinterpret_cast<uint64_t>(బఫరు);
        response.remote_కొలత = బఫరు->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_కొలత(request.size).c_str(), response.remote_ptr,
                rpc_log_format_కొలత(response.remote_కొలత).c_str());
        buffers.insert(బఫరు);
    } లేకపోతే {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_కొలత(request.size).c_str());
    }
    తిరిగి నిజం;
}

తార్కిక rpc_server::get_alignment(స్థిరరాశి rpc_సందేశం_get_alignment_req & request, rpc_సందేశం_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    ఒకవేళ (dev_id >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    ggml_backend_బఫరు_type_t buft = ggml_backend_get_default_బఫరు_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_కొలత(alignment).c_str());
    response.alignment = alignment;
    తిరిగి నిజం;
}

తార్కిక rpc_server::get_max_కొలత(స్థిరరాశి rpc_సందేశం_get_max_కొలత_req & request, rpc_సందేశం_get_max_కొలత_rsp & response) {
    uint32_t dev_id = request.device;
    ఒకవేళ (dev_id >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    ggml_backend_బఫరు_type_t buft = ggml_backend_get_default_బఫరు_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_కొలత(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_కొలత(max_size).c_str());
    response.max_size = max_size;
    తిరిగి నిజం;
}

తార్కిక rpc_server::బఫరు_get_పునాది(స్థిరరాశి rpc_సందేశం_బఫరు_get_పునాది_req & request, rpc_సందేశం_బఫరు_get_పునాది_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_బఫరు_t బఫరు = reinterpret_cast<ggml_backend_బఫరు_t>(request.remote_ptr);
    ఒకవేళ (buffers.find(బఫరు) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        తిరిగి అబద్ధం;
    }
    శూన్య * పునాది = ggml_backend_బఫరు_get_పునాది(బఫరు);
    response.పునాది_ptr = reinterpret_cast<uint64_t>(పునాది);
    తిరిగి నిజం;
}

తార్కిక rpc_server::free_బఫరు(స్థిరరాశి rpc_సందేశం_free_బఫరు_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_బఫరు_t బఫరు = reinterpret_cast<ggml_backend_బఫరు_t>(request.remote_ptr);
    ఒకవేళ (buffers.find(బఫరు) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        తిరిగి అబద్ధం;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    కొరకు (స్వయంచాలక & sgs : stored_graphs) {
        కొరకు (స్వయంచాలక & sg : sgs) {
            sg.second.graph = శూన్యసూచిక;
        }
    }
    ggml_backend_బఫరు_free(బఫరు);
    buffers.erase(బఫరు);
    తిరిగి నిజం;
}

తార్కిక rpc_server::బఫరు_clear(స్థిరరాశి rpc_సందేశం_బఫరు_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_బఫరు_t బఫరు = reinterpret_cast<ggml_backend_బఫరు_t>(request.remote_ptr);
    ఒకవేళ (buffers.find(బఫరు) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        తిరిగి అబద్ధం;
    }
    ggml_backend_బఫరు_clear(బఫరు, request.value);
    తిరిగి నిజం;
}

తార్కిక rpc_server::memset_tensor(స్థిరరాశి rpc_సందేశం_memset_tensor_req & request) {
    sync_all_backends();
    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ఒకవేళ (tensor == శూన్యసూచిక || tensor->బఫరు == శూన్యసూచిక) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint64_t tensor_కొలత = ggml_nbytes(tensor);
    ఒకవేళ (request.స్థానభ్రంశం > tensor_కొలత || request.size > tensor_కొలత - request.స్థానభ్రంశం) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.స్థానభ్రంశం, request.size, tensor_కొలత);
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint64_t బఫరు_ప్రారంభం = (uint64_t) ggml_backend_బఫరు_get_పునాది(tensor->బఫరు);
    స్థిరరాశి uint64_t బఫరు_కొలత = ggml_backend_బఫరు_get_కొలత(tensor->బఫరు);
    ఒకవేళ (request.tensor.దత్తాంశం < బఫరు_ప్రారంభం) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి uint64_t దత్తాంశం_స్థానభ్రంశం = request.tensor.దత్తాంశం - బఫరు_ప్రారంభం;
    ఒకవేళ (దత్తాంశం_స్థానభ్రంశం > బఫరు_కొలత ||
        request.స్థానభ్రంశం > బఫరు_కొలత - దత్తాంశం_స్థానభ్రంశం ||
        request.size > బఫరు_కొలత - దత్తాంశం_స్థానభ్రంశం - request.స్థానభ్రంశం) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (tensor->బఫరు->iface.memset_tensor == శూన్యసూచిక) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        తిరిగి అబద్ధం;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (శూన్య *) tensor->బఫరు, tensor->దత్తాంశం, request.స్థానభ్రంశం,
            rpc_log_format_కొలత(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.స్థానభ్రంశం, request.size);
    తిరిగి నిజం;
}

ggml_tensor * rpc_server::deserialize_tensor(నిర్మాణం ggml_context * ctx, స్థిరరాశి rpc_tensor * tensor) {
    // Validate tensor type before using it
    ఒకవేళ (tensor->type >= GGML_TYPE_లెక్క) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        తిరిగి శూన్యసూచిక;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    ఒకవేళ (ggml_blck_కొలత((గణన ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        తిరిగి శూన్యసూచిక;
    }

    ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    ఒకవేళ (ఫలితం == శూన్యసూచిక) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        తిరిగి శూన్యసూచిక;
    }

    కొరకు (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        ఫలితం->nb[i] = tensor->nb[i];
    }
    ఫలితం->బఫరు = reinterpret_cast<ggml_backend_బఫరు_t>(tensor->బఫరు);
    ఒకవేళ (ఫలితం->బఫరు && buffers.find(ఫలితం->బఫరు) == buffers.end()) {
        ఫలితం->బఫరు = శూన్యసూచిక;
    }

    ఒకవేళ (ఫలితం->బఫరు && ggml_nelements(ఫలితం) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_కొలత = (uint64_t) ggml_nbytes(ఫలితం);
        uint64_t బఫరు_ప్రారంభం = (uint64_t) ggml_backend_బఫరు_get_పునాది(ఫలితం->బఫరు);
        uint64_t బఫరు_కొలత = (uint64_t) ggml_backend_బఫరు_get_కొలత(ఫలితం->బఫరు);
        ఒకవేళ (tensor->దత్తాంశం + tensor_కొలత < tensor->దత్తాంశం ||
            tensor->దత్తాంశం < బఫరు_ప్రారంభం || tensor->దత్తాంశం + tensor_కొలత > బఫరు_ప్రారంభం + బఫరు_కొలత) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(ఫలితం->type),
                      ఫలితం->ne[0], ఫలితం->ne[1], ఫలితం->ne[2], ఫలితం->ne[3],
                      tensor->దత్తాంశం, tensor->దత్తాంశం + tensor_కొలత, బఫరు_ప్రారంభం, బఫరు_ప్రారంభం + బఫరు_కొలత);
            తిరిగి శూన్యసూచిక;
        }
    }

    ఫలితం->op = (ggml_op) tensor->op;
    కొరకు (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / పరిమాణం(int32_t); i++) {
        ఫలితం->op_params[i] = tensor->op_params[i];
    }
    ఫలితం->flags = tensor->flags;
    ఫలితం->దత్తాంశం = reinterpret_cast<శూన్య *>(tensor->దత్తాంశం);
    ggml_సమితి_name(ఫలితం, tensor->name);
    తిరిగి ఫలితం;
}


తార్కిక rpc_server::సమితి_tensor(స్థిరరాశి std::vector<uint8_t> & ఆదానం) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t స్థానభ్రంశం;
    స్థిరరాశి size_t శీర్షిక_కొలత = పరిమాణం(rpc_tensor) + పరిమాణం(cache_flag) + పరిమాణం(స్థానభ్రంశం);
    ఒకవేళ (ఆదానం.size() < శీర్షిక_కొలత) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి rpc_tensor * లోపల_tensor = (స్థిరరాశి rpc_tensor *)ఆదానం.దత్తాంశం();
    జ్ఞాపకం_నకలు(&cache_flag, ఆదానం.దత్తాంశం() + పరిమాణం(rpc_tensor), పరిమాణం(cache_flag));
    జ్ఞాపకం_నకలు(&స్థానభ్రంశం,     ఆదానం.దత్తాంశం() + పరిమాణం(rpc_tensor) + పరిమాణం(cache_flag), పరిమాణం(స్థానభ్రంశం));
    స్థిరరాశి size_t size = ఆదానం.size() - శీర్షిక_కొలత;

    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, లోపల_tensor);
    ఒకవేళ (tensor == శూన్యసూచిక || tensor->బఫరు == శూన్యసూచిక) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        తిరిగి అబద్ధం;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (శూన్య*)tensor->బఫరు, tensor->దత్తాంశం, స్థానభ్రంశం,
            rpc_log_format_కొలత(size).c_str());

    // sanitize tensor->data
    {
        స్థిరరాశి size_t p0 = (size_t) ggml_backend_బఫరు_get_పునాది(tensor->బఫరు);
        స్థిరరాశి size_t p1 = p0 + ggml_backend_బఫరు_get_కొలత(tensor->బఫరు);

        ఒకవేళ (లోపల_tensor->దత్తాంశం + స్థానభ్రంశం < p0 || లోపల_tensor->దత్తాంశం + స్థానభ్రంశం >= p1 || size > (p1 - లోపల_tensor->దత్తాంశం - స్థానభ్రంశం)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, లోపల_tensor->దత్తాంశం, స్థానభ్రంశం, size, p0, p1);
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి శూన్య * దత్తాంశం = ఆదానం.దత్తాంశం() + శీర్షిక_కొలత;
    ఒకవేళ (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((స్థిరరాశి uint8_t*)దత్తాంశం, size);
        అక్షరం hash_str[17];
        snprintf(hash_str, పరిమాణం(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.లిఖ((స్థిరరాశి అక్షరం *)దత్తాంశం, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_సమితి(tensor, దత్తాంశం, స్థానభ్రంశం, size);
    తిరిగి నిజం;
}

తార్కిక rpc_server::సమితి_tensor_2d(స్థిరరాశి std::vector<uint8_t> & ఆదానం) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    ఒకవేళ (ఆదానం.size() < పరిమాణం(rpc_tensor) + 4*పరిమాణం(uint64_t)) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి rpc_tensor * లోపల_tensor = (స్థిరరాశి rpc_tensor *)ఆదానం.దత్తాంశం();
    uint64_t శీర్షిక[4];
    జ్ఞాపకం_నకలు(శీర్షిక, ఆదానం.దత్తాంశం() + పరిమాణం(rpc_tensor), పరిమాణం(శీర్షిక));
    స్థిరరాశి uint64_t స్థానభ్రంశం   = శీర్షిక[0];
    స్థిరరాశి uint64_t size     = శీర్షిక[1];
    స్థిరరాశి uint64_t n_copies = శీర్షిక[2];
    స్థిరరాశి uint64_t stride   = శీర్షిక[3];

    స్థిరరాశి uint64_t దత్తాంశం_కొలత = ఆదానం.size() - పరిమాణం(rpc_tensor) - 4*పరిమాణం(uint64_t);
    ఒకవేళ (n_copies == 0 || size == 0 || size > దత్తాంశం_కొలత / n_copies || size * n_copies != దత్తాంశం_కొలత) {
        తిరిగి అబద్ధం;
    }

    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, లోపల_tensor);
    ఒకవేళ (tensor == శూన్యసూచిక || tensor->బఫరు == శూన్యసూచిక) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        తిరిగి అబద్ధం;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (శూన్య*)tensor->బఫరు, tensor->దత్తాంశం, స్థానభ్రంశం,
            rpc_log_format_కొలత(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        ఒకవేళ (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి uint64_t పరిధి = (n_copies - 1)*stride + size;
        స్థిరరాశి uint64_t p0 = (uint64_t) ggml_backend_బఫరు_get_పునాది(tensor->బఫరు);
        స్థిరరాశి uint64_t p1 = p0 + ggml_backend_బఫరు_get_కొలత(tensor->బఫరు);

        ఒకవేళ (లోపల_tensor->దత్తాంశం < p0 || లోపల_tensor->దత్తాంశం > p1 || స్థానభ్రంశం > p1 - లోపల_tensor->దత్తాంశం || పరిధి > p1 - లోపల_tensor->దత్తాంశం - స్థానభ్రంశం) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, లోపల_tensor->దత్తాంశం, స్థానభ్రంశం, పరిధి, p0, p1);
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (స్థానభ్రంశం > ggml_nbytes(tensor) || పరిధి > ggml_nbytes(tensor) - స్థానభ్రంశం) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, స్థానభ్రంశం, పరిధి, ggml_nbytes(tensor));
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి శూన్య * దత్తాంశం = ఆదానం.దత్తాంశం() + పరిమాణం(rpc_tensor) + 4*పరిమాణం(uint64_t);
    ggml_backend_tensor_సమితి_2d(tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride, size);
    తిరిగి నిజం;
}

తార్కిక rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & దత్తాంశం) {
    ఒకవేళ (!cache_dir) {
        తిరిగి అబద్ధం;
    }
    అక్షరం hash_str[17];
    snprintf(hash_str, పరిమాణం(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    ఒకవేళ (!fs::exists(cache_file, ec)) {
        తిరిగి అబద్ధం;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    దత్తాంశం.resize(size);
    ifs.పఠ((అక్షరం *)దత్తాంశం.దత్తాంశం(), size);
    తిరిగి నిజం;
}

తార్కిక rpc_server::సమితి_tensor_hash(స్థిరరాశి rpc_సందేశం_సమితి_tensor_hash_req & request, rpc_సందేశం_సమితి_tensor_hash_rsp & response)
{
    sync_all_backends();
    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ఒకవేళ (tensor == శూన్యసూచిక || tensor->బఫరు == శూన్యసూచిక) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        తిరిగి అబద్ధం;
    }

    std::vector<uint8_t> cached_file;
    ఒకవేళ (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.స్థానభ్రంశం, request.hash);
        response.ఫలితం = 0;
        తిరిగి నిజం;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (శూన్య*)tensor->బఫరు, tensor->దత్తాంశం, request.స్థానభ్రంశం,
            rpc_log_format_కొలత(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        స్థిరరాశి size_t p0 = (size_t) ggml_backend_బఫరు_get_పునాది(tensor->బఫరు);
        స్థిరరాశి size_t p1 = p0 + ggml_backend_బఫరు_get_కొలత(tensor->బఫరు);

        ఒకవేళ (request.tensor.దత్తాంశం + request.స్థానభ్రంశం < p0
         || request.tensor.దత్తాంశం + request.స్థానభ్రంశం >= p1
         || size > (p1 - request.tensor.దత్తాంశం - request.స్థానభ్రంశం)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.దత్తాంశం, request.స్థానభ్రంశం, size, request.hash, p0, p1);
            తిరిగి అబద్ధం;
        }
    }
    ggml_backend_tensor_సమితి(tensor, cached_file.దత్తాంశం(), request.స్థానభ్రంశం, size);
    response.ఫలితం = 1;
    తిరిగి నిజం;
}

తార్కిక rpc_server::init_tensor(స్థిరరాశి rpc_సందేశం_init_tensor_req & request) {
    sync_all_backends();
    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ఒకవేళ (tensor == శూన్యసూచిక) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        తిరిగి అబద్ధం;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (శూన్య*)tensor->బఫరు, tensor->దత్తాంశం);
    // Call the backend's buffer_init_tensor function
    ggml_backend_బఫరు_t బఫరు = tensor->బఫరు;
    ఒకవేళ (బఫరు && బఫరు->iface.init_tensor) {
        బఫరు->iface.init_tensor(బఫరు, tensor);
    } లేకపోతే {
        ఒకవేళ (!బఫరు) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    ఒకవేళ (tensor->extra != శూన్యసూచిక) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

తార్కిక rpc_server::get_tensor(స్థిరరాశి rpc_సందేశం_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ఒకవేళ (tensor == శూన్యసూచిక || tensor->బఫరు == శూన్యసూచిక) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        తిరిగి అబద్ధం;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (శూన్య*)tensor->బఫరు, tensor->దత్తాంశం, request.స్థానభ్రంశం,
             rpc_log_format_కొలత(request.size).c_str());

    // sanitize tensor->data
    {
        స్థిరరాశి size_t p0 = (size_t) ggml_backend_బఫరు_get_పునాది(tensor->బఫరు);
        స్థిరరాశి size_t p1 = p0 + ggml_backend_బఫరు_get_కొలత(tensor->బఫరు);

        ఒకవేళ (request.tensor.దత్తాంశం + request.స్థానభ్రంశం < p0 ||
            request.tensor.దత్తాంశం + request.స్థానభ్రంశం >= p1 ||
            request.size > (p1 - request.tensor.దత్తాంశం - request.స్థానభ్రంశం)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.దత్తాంశం, request.స్థానభ్రంశం, request.size, p0, p1);
                తిరిగి అబద్ధం;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.దత్తాంశం(), request.స్థానభ్రంశం, request.size);
    తిరిగి నిజం;
}

తార్కిక rpc_server::get_tensor_2d(స్థిరరాశి rpc_సందేశం_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    ఒకవేళ (tensor == శూన్యసూచిక || tensor->బఫరు == శూన్యసూచిక) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        తిరిగి అబద్ధం;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (శూన్య*)tensor->బఫరు, tensor->దత్తాంశం, request.స్థానభ్రంశం,
            rpc_log_format_కొలత(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        ఒకవేళ (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి uint64_t పరిధి = (request.n_copies - 1)*request.stride + request.size;
        స్థిరరాశి uint64_t p0 = (uint64_t) ggml_backend_బఫరు_get_పునాది(tensor->బఫరు);
        స్థిరరాశి uint64_t p1 = p0 + ggml_backend_బఫరు_get_కొలత(tensor->బఫరు);

        ఒకవేళ (request.tensor.దత్తాంశం < p0 || request.tensor.దత్తాంశం > p1 || request.స్థానభ్రంశం > p1 - request.tensor.దత్తాంశం ||
                పరిధి > p1 - request.tensor.దత్తాంశం - request.స్థానభ్రంశం) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.దత్తాంశం, request.స్థానభ్రంశం, పరిధి, p0, p1);
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (request.స్థానభ్రంశం > ggml_nbytes(tensor) || పరిధి > ggml_nbytes(tensor) - request.స్థానభ్రంశం) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.స్థానభ్రంశం, పరిధి, ggml_nbytes(tensor));
            తిరిగి అబద్ధం;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.దత్తాంశం(), request.స్థానభ్రంశం, request.size, request.n_copies, request.stride, request.size);
    తిరిగి నిజం;
}

తార్కిక rpc_server::copy_tensor(స్థిరరాశి rpc_సందేశం_copy_tensor_req & request, rpc_సందేశం_copy_tensor_rsp & response) {
    sync_all_backends();
    నిర్మాణం ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    ఒకవేళ (src == శూన్యసూచిక || dst == శూన్యసూచిక || src->బఫరు == శూన్యసూచిక || dst->బఫరు == శూన్యసూచిక) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        తిరిగి అబద్ధం;
    }

    uint64_t src_కొలత   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_దత్తాంశం   = (uint64_t) dst->దత్తాంశం;
    uint64_t dst_పునాది   = (uint64_t) ggml_backend_బఫరు_get_పునాది(dst->బఫరు);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_బఫరు_get_కొలత(dst->బఫరు);

    ఒకవేళ (dst_దత్తాంశం + src_కొలత > dst_పునాది + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_దత్తాంశం,
                         dst_దత్తాంశం + src_కొలత,
                         dst_పునాది,
                         dst_పునాది + dst_buf_sz);
        తిరిగి అబద్ధం;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.ఫలితం = ggml_backend_బఫరు_copy_tensor(src, dst);
    తిరిగి నిజం;
}

ggml_tensor * rpc_server::create_కణుపు(uint64_t id,
                                      నిర్మాణం ggml_context * ctx,
                                      స్థిరరాశి std::unordered_map<uint64_t, స్థిరరాశి rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, నిర్మాణం ggml_tensor*> & tensor_map) {
    ఒకవేళ (tensor_map.find(id) != tensor_map.end()) {
        తిరిగి tensor_map[id];
    }
    // Safely find the tensor pointer
    స్వయంచాలక it_ptr = tensor_ptrs.find(id);
    ఒకవేళ (it_ptr == tensor_ptrs.end()) {
        తిరిగి శూన్యసూచిక;
    }
    స్థిరరాశి rpc_tensor * tensor = it_ptr->second;

    నిర్మాణం ggml_tensor * ఫలితం = deserialize_tensor(ctx, tensor);
    ఒకవేళ (ఫలితం == శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }
    ఒకవేళ (ఫలితం->బఫరు == శూన్యసూచిక && ఫలితం->దత్తాంశం != శూన్యసూచిక) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        తిరిగి శూన్యసూచిక;
    }
    tensor_map[id] = ఫలితం;
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        ఒకవేళ (tensor->src[i] == 0) {
            ఫలితం->src[i] = శూన్యసూచిక;
        } లేకపోతే {
            ఫలితం->src[i] = create_కణుపు(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            ఒకవేళ (ఫలితం->src[i] == శూన్యసూచిక) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                తిరిగి శూన్యసూచిక;
            }
        }
    }

    // Handle view_src similarly
    ఒకవేళ (tensor->view_src == 0) {
        ఫలితం->view_src = శూన్యసూచిక;
    } లేకపోతే {
        ఫలితం->view_src = create_కణుపు(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        ఒకవేళ (ఫలితం->view_src == శూన్యసూచిక) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            తిరిగి శూన్యసూచిక;
        }
    }
    ఫలితం->view_offs = tensor->view_offs;
    తిరిగి ఫలితం;
}

తార్కిక rpc_server::లేఖాచిత్రం_compute(స్థిరరాశి std::vector<uint8_t> & ఆదానం) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    ఒకవేళ (ఆదానం.size() < 2*పరిమాణం(uint32_t) + పరిమాణం(uint64_t)) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి uint8_t * src = ఆదానం.దత్తాంశం();
    uint32_t device;
    జ్ఞాపకం_నకలు(&device, src, పరిమాణం(device));
    src += పరిమాణం(device);
    ఒకవేళ (device >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    uint64_t uid;
    జ్ఞాపకం_నకలు(&uid, src, పరిమాణం(uid));
    src += పరిమాణం(uid);
    uint32_t n_nodes;
    జ్ఞాపకం_నకలు(&n_nodes, src, పరిమాణం(n_nodes));
    src += పరిమాణం(n_nodes);
    ఒకవేళ (ఆదానం.size() < 2*పరిమాణం(uint32_t) + పరిమాణం(uint64_t) + n_nodes*పరిమాణం(uint64_t) + పరిమాణం(uint32_t)) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి uint64_t * nodes = (స్థిరరాశి uint64_t *)src;
    src += n_nodes*పరిమాణం(uint64_t);
    uint32_t n_tensors;
    జ్ఞాపకం_నకలు(&n_tensors, src, పరిమాణం(n_tensors));
    src += పరిమాణం(n_tensors);
    ఒకవేళ (ఆదానం.size() < 2*పరిమాణం(uint32_t) + పరిమాణం(uint64_t) + n_nodes*పరిమాణం(uint64_t) + పరిమాణం(uint32_t) + n_tensors*పరిమాణం(rpc_tensor)) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి rpc_tensor * tensors = (స్థిరరాశి rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    ఒకవేళ (uid != 0 && stored_graphs[device].size() >= లేఖాచిత్రం_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_లేఖాచిత్రం sg_tmp;
    stored_లేఖాచిత్రం & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_కొలత = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_లేఖాచిత్రం_overhead_custom(n_nodes, అబద్ధం);
    ఒకవేళ (sg.బఫరు.size() < buf_కొలత) {
        sg.బఫరు.resize(buf_కొలత);
    }
    నిర్మాణం ggml_init_params params = {
        /*.mem_size   =*/ buf_కొలత,
        /*.mem_buffer =*/ sg.బఫరు.దత్తాంశం(),
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    నిర్మాణం ggml_cgraph * graph = ggml_new_లేఖాచిత్రం_custom(ctx, n_nodes, అబద్ధం);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, స్థిరరాశి rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    కొరకు (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    స్థిరరాశి స్వయంచాలక t_deserialize = std::chrono::steady_clock::now();
    కొరకు (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        జ్ఞాపకం_నకలు(&id, &nodes[i], పరిమాణం(id));
        graph->nodes[i] = create_కణుపు(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        ఒకవేళ (graph->nodes[i] == శూన్యసూచిక && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (graph->nodes[i] != శూన్యసూచిక) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            స్థిరరాశి size_t hash_pos = ggml_hash_చొప్పించు(&graph->సందర్శించిన_hash_సమితి, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_లెక్క;
        }
    }
    స్థిరరాశి std::chrono::duration<ద్విగుణ, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_స్థితి స్థితి_2 = ggml_backend_లేఖాచిత్రం_compute_async(backends[device], graph);
    GGML_ASSERT(స్థితి_2 == GGML_స్థితి_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    తిరిగి నిజం;
}

తార్కిక rpc_server::లేఖాచిత్రం_recompute(స్థిరరాశి rpc_సందేశం_లేఖాచిత్రం_recompute_req & request) {
    uint32_t device = request.device;
    ఒకవేళ (device >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    స్వయంచాలక it = stored_graphs[device].find(request.uid);
    ఒకవేళ (it == stored_graphs[device].end() || it->second.graph == శూన్యసూచిక) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        తిరిగి అబద్ధం;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_స్థితి స్థితి_2 = ggml_backend_లేఖాచిత్రం_compute_async(backends[device], graph);
    GGML_ASSERT(స్థితి_2 == GGML_స్థితి_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    తిరిగి నిజం;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
శూన్య rpc_server::sync_all_backends() {
    కొరకు (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
తార్కిక rpc_server::comm_init(స్థిరరాశి rpc_సందేశం_comm_init_req & request, rpc_సందేశం_comm_init_rsp & response) {
    response.ok = 0;
    ఒకవేళ (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        తిరిగి నిజం;
    }
    comm_state & state = comm_స్థితులు[request.device];
    ఒకవేళ (state.peer != శూన్యసూచిక) {
        response.ok = 1;
        తిరిగి నిజం;
    }
    uint8_t local_caps[RPC_CONN_CAPS_కొలత] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_కొలత] = {};
    ఒకవేళ (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.పోర్టు);
        ఒకవేళ (srv == శూన్యసూచిక) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.పోర్టు);
            తిరిగి నిజం;
        }
        state.peer = srv->accept();
        ఒకవేళ (state.peer == శూన్యసూచిక) {
            తిరిగి నిజం;
        }
        ఒకవేళ (!state.peer->recv_దత్తాంశం(remote_caps, పరిమాణం(remote_caps))) {
            state.peer = శూన్యసూచిక;
            తిరిగి నిజం;
        }
        state.peer->get_caps(local_caps);
        ఒకవేళ (!state.peer->send_దత్తాంశం(local_caps, పరిమాణం(local_caps))) {
            state.peer = శూన్యసూచిక;
            తిరిగి నిజం;
        }
        state.peer->update_caps(remote_caps);
    } లేకపోతే {
        స్థిరరాశి std::string host(request.host, strnlen(request.host, పరిమాణం(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        కొరకు (పూర్ణాంకం i = 0; i < 100 && state.peer == శూన్యసూచిక; i++) {
            state.peer = socket_t::connect(host.c_str(), request.పోర్టు);
            ఒకవేళ (state.peer == శూన్యసూచిక) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        ఒకవేళ (state.peer == శూన్యసూచిక) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.పోర్టు);
            తిరిగి నిజం;
        }
        state.peer->get_caps(local_caps);
        ఒకవేళ (!state.peer->send_దత్తాంశం(local_caps, పరిమాణం(local_caps)) ||
            !state.peer->recv_దత్తాంశం(remote_caps, పరిమాణం(remote_caps))) {
            state.peer = శూన్యసూచిక;
            తిరిగి నిజం;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    తిరిగి నిజం;
}

తార్కిక rpc_server::comm_allreduce(స్థిరరాశి rpc_సందేశం_comm_allreduce_req & request) {
    ఒకవేళ (request.device >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    comm_state & state = comm_స్థితులు[request.device];
    ఒకవేళ (state.peer == శూన్యసూచిక) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        తిరిగి అబద్ధం;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_కొలత = 16*ggml_tensor_overhead() + 2*ggml_లేఖాచిత్రం_overhead_custom(8, అబద్ధం);
    నిర్మాణం ggml_init_params params = {
        /*.mem_size   =*/ ctx_కొలత,
        /*.mem_buffer =*/ ఖాళీ,
        /*.no_alloc   =*/ నిజం,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != శూన్యసూచిక);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    ఒకవేళ (t_dst == శూన్యసూచిక || t_dst->బఫరు == శూన్యసూచిక) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి size_t  nbytes = ggml_nbytes(t_dst);
    స్థిరరాశి int64_t ne     = ggml_nelements(t_dst);
    ఒకవేళ (nbytes == 0) {
        తిరిగి నిజం;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    స్థిరరాశి తార్కిక   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    స్థిరరాశి size_t wire_బైట్లు = wire_bf16 ? (size_t) ne*2 : nbytes;
    స్థిరరాశి size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    ఒకవేళ (state.scratch_కొలత < need) {
        state.scratch.reset(ggml_backend_alloc_బఫరు(backend, need));
        state.scratch_కొలత = need;
    }
    అక్షరం * scratch_పునాది = (అక్షరం *) ggml_backend_బఫరు_get_పునాది(state.scratch.get());
    state.send_buf.resize(wire_బైట్లు);
    state.recv_buf.resize(wire_బైట్లు);

    స్వయంచాలక new_scratch_tensor = [&](ggml_type type, size_t స్థానభ్రంశం) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->బఫరు = state.scratch.get();
        t->దత్తాంశం   = scratch_పునాది + స్థానభ్రంశం;
        తిరిగి t;
    };
    స్వయంచాలక new_cpy_కణుపు = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->బఫరు = dst->బఫరు;
        t->దత్తాంశం   = dst->దత్తాంశం;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        తిరిగి t;
    };
    స్వయంచాలక compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_లేఖాచిత్రం_custom(ctx, 2, అబద్ధం);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != శూన్యసూచిక ? 2 : 1;
        ggml_స్థితి స్థితి_2 = ggml_backend_లేఖాచిత్రం_compute_async(backend, graph);
        GGML_ASSERT(స్థితి_2 == GGML_స్థితి_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = శూన్యసూచిక;
    ggml_tensor * t_wire_recv = శూన్యసూచిక;
    ఒకవేళ (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_కణుపు(t_dst, t_wire_send), శూన్యసూచిక);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.దత్తాంశం(), 0, wire_బైట్లు);
    } లేకపోతే {
        ggml_backend_tensor_get(t_dst, state.send_buf.దత్తాంశం(), 0, wire_బైట్లు);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    ఒకవేళ (state.rank == 0) {
        ఒకవేళ (!state.peer->send_దత్తాంశం(state.send_buf.దత్తాంశం(), wire_బైట్లు) || !state.peer->flush() ||
            !state.peer->recv_దత్తాంశం(state.recv_buf.దత్తాంశం(), wire_బైట్లు)) {
            తిరిగి అబద్ధం;
        }
    } లేకపోతే {
        ఒకవేళ (!state.peer->recv_దత్తాంశం(state.recv_buf.దత్తాంశం(), wire_బైట్లు) ||
            !state.peer->send_దత్తాంశం(state.send_buf.దత్తాంశం(), wire_బైట్లు) || !state.peer->flush()) {
            తిరిగి అబద్ధం;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = శూన్యసూచిక;
    ఒకవేళ (wire_bf16) {
        ggml_backend_tensor_సమితి(t_wire_recv, state.recv_buf.దత్తాంశం(), 0, wire_బైట్లు);
        t_cast = new_cpy_కణుపు(t_wire_recv, t_peer);
    } లేకపోతే {
        ggml_backend_tensor_సమితి(t_peer, state.recv_buf.దత్తాంశం(), 0, wire_బైట్లు);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_కూడు;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->బఫరు = t_dst->బఫరు;
    t_red->దత్తాంశం   = t_dst->దత్తాంశం;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    ఒకవేళ (t_cast != శూన్యసూచిక) {
        compute_nodes(t_cast, t_red);
    } లేకపోతే {
        compute_nodes(t_red, శూన్యసూచిక);
    }
    తిరిగి నిజం;
}

తార్కిక rpc_server::comm_free(స్థిరరాశి rpc_సందేశం_comm_free_req & request) {
    ఒకవేళ (request.device >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    comm_స్థితులు[request.device] = comm_state();
    తిరిగి నిజం;
}

తార్కిక rpc_server::get_device_జ్ఞాపకం(స్థిరరాశి rpc_సందేశం_get_device_జ్ఞాపకం_req & request, rpc_సందేశం_get_device_జ్ఞాపకం_rsp & response) {
    uint32_t dev_id = request.device;
    ఒకవేళ (dev_id >= backends.size()) {
        తిరిగి అబద్ధం;
    }
    size_t విడుదల, మొత్తం;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_జ్ఞాపకం(dev, &విడుదల, &మొత్తం);
    response.free_mem = విడుదల;
    response.మొత్తం_mem = మొత్తం;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_కొలత(response.free_mem).c_str(), rpc_log_format_కొలత(response.మొత్తం_mem).c_str());
    తిరిగి నిజం;
}

rpc_server::~rpc_server() {
    కొరకు (స్వయంచాలక బఫరు : buffers) {
        ggml_backend_బఫరు_free(బఫరు);
    }
}

స్థిర శూన్య rpc_serve_client(స్థిరరాశి std::vector<ggml_backend_t> & backends, స్థిరరాశి అక్షరం * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    ఒకవేళ (!sock->recv_దత్తాంశం(&cmd, 1)) {
        తిరిగి;
    }
    ఒకవేళ (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        తిరిగి;
    }

    // Read input_size and validate protocol version
    uint64_t hello_ఆదానం_కొలత;
    ఒకవేళ (!sock->recv_దత్తాంశం(&hello_ఆదానం_కొలత, పరిమాణం(hello_ఆదానం_కొలత))) {
        తిరిగి;
    }

    ఒకవేళ (hello_ఆదానం_కొలత != పరిమాణం(rpc_సందేశం_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_ఆదానం_కొలత, పరిమాణం(rpc_సందేశం_hello_req), RPC_PROTO_MAJOR_సంస్కరణ);
        తిరిగి;
    }

    rpc_సందేశం_hello_req req = {};
    ఒకవేళ (!sock->recv_దత్తాంశం(&req, పరిమాణం(req))) {
        తిరిగి;
    }

    rpc_సందేశం_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    ఒకవేళ (!send_సందేశం(sock, &rsp, పరిమాణం(rsp))) {
        తిరిగి;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    అయితే (నిజం) {
        ఒకవేళ (!sock->recv_దత్తాంశం(&cmd, 1)) {
            ఆపు;
        }
        ఒకవేళ (cmd >= RPC_CMD_లెక్క) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            ఆపు;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((గణన rpc_cmd)cmd));
        ఎంపిక (cmd) {
            స్థితి RPC_CMD_HELLO: {
                // HELLO command is handled above
                తిరిగి;
            }
            స్థితి RPC_CMD_DEVICE_లెక్క: {
                ఒకవేళ (!recv_సందేశం(sock, శూన్యసూచిక, 0)) {
                    తిరిగి;
                }
                rpc_సందేశం_device_లెక్క_rsp response;
                response.device_లెక్క = backends.size();
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_ALLOC_బఫరు: {
                rpc_సందేశం_alloc_బఫరు_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_alloc_బఫరు_rsp response;
                ఒకవేళ (!server.alloc_బఫరు(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_GET_ALLOC_కొలత: {
                rpc_సందేశం_get_alloc_కొలత_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_get_alloc_కొలత_rsp response;
                ఒకవేళ (!server.get_alloc_కొలత(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_GET_ALIGNMENT: {
                rpc_సందేశం_get_alignment_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_get_alignment_rsp response;
                ఒకవేళ (!server.get_alignment(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_GET_MAX_కొలత: {
                rpc_సందేశం_get_max_కొలత_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_get_max_కొలత_rsp response;
                ఒకవేళ (!server.get_max_కొలత(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_బఫరు_GET_పునాది: {
                rpc_సందేశం_బఫరు_get_పునాది_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_బఫరు_get_పునాది_rsp response;
                ఒకవేళ (!server.బఫరు_get_పునాది(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_FREE_బఫరు: {
                rpc_సందేశం_free_బఫరు_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                ఒకవేళ (!server.free_బఫరు(request)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_బఫరు_CLEAR: {
                rpc_సందేశం_బఫరు_clear_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                ఒకవేళ (!server.బఫరు_clear(request)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_MEMSET_TENSOR: {
                rpc_సందేశం_memset_tensor_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                ఒకవేళ (!server.memset_tensor(request)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_సమితి_TENSOR: {
                std::vector<uint8_t> ఆదానం;
                ఒకవేళ (!recv_సందేశం(sock, ఆదానం)) {
                    తిరిగి;
                }
                ఒకవేళ (!server.సమితి_tensor(ఆదానం)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_సమితి_TENSOR_2_D: {
                std::vector<uint8_t> ఆదానం;
                ఒకవేళ (!recv_సందేశం(sock, ఆదానం)) {
                    తిరిగి;
                }
                ఒకవేళ (!server.సమితి_tensor_2d(ఆదానం)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_GET_TENSOR_2D: {
                rpc_సందేశం_get_tensor_2d_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                std::vector<uint8_t> response;
                ఒకవేళ (!server.get_tensor_2d(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, response.దత్తాంశం(), response.size())) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_సమితి_TENSOR_HASH: {
                rpc_సందేశం_సమితి_tensor_hash_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_సమితి_tensor_hash_rsp response;
                ఒకవేళ (!server.సమితి_tensor_hash(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_INIT_TENSOR: {
                rpc_సందేశం_init_tensor_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request,పరిమాణం(request))) {
                    తిరిగి;
                }
                ఒకవేళ (!server.init_tensor(request)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_GET_TENSOR: {
                rpc_సందేశం_get_tensor_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                std::vector<uint8_t> response;
                ఒకవేళ (!server.get_tensor(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, response.దత్తాంశం(), response.size())) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_COPY_TENSOR: {
                rpc_సందేశం_copy_tensor_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_copy_tensor_rsp response;
                ఒకవేళ (!server.copy_tensor(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_లేఖాచిత్రం_COMPUTE: {
                std::vector<uint8_t> ఆదానం;
                ఒకవేళ (!recv_సందేశం(sock, ఆదానం)) {
                    తిరిగి;
                }
                ఒకవేళ (!server.లేఖాచిత్రం_compute(ఆదానం)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_లేఖాచిత్రం_RECOMPUTE: {
                rpc_సందేశం_లేఖాచిత్రం_recompute_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                ఒకవేళ (!server.లేఖాచిత్రం_recompute(request)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_COMM_INIT: {
                rpc_సందేశం_comm_init_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_comm_init_rsp response;
                ఒకవేళ (!server.comm_init(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_COMM_ALLREDUCE: {
                rpc_సందేశం_comm_allreduce_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                ఒకవేళ (!server.comm_allreduce(request)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_COMM_FREE: {
                rpc_సందేశం_comm_free_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                ఒకవేళ (!server.comm_free(request)) {
                    తిరిగి;
                }
                ఆపు;
            }
            స్థితి RPC_CMD_GET_DEVICE_జ్ఞాపకం: {
                rpc_సందేశం_get_device_జ్ఞాపకం_req request;
                ఒకవేళ (!recv_సందేశం(sock, &request, పరిమాణం(request))) {
                    తిరిగి;
                }
                rpc_సందేశం_get_device_జ్ఞాపకం_rsp response;
                ఒకవేళ (!server.get_device_జ్ఞాపకం(request, response)) {
                    తిరిగి;
                }
                ఒకవేళ (!send_సందేశం(sock, &response, పరిమాణం(response))) {
                    తిరిగి;
                }
                ఆపు;
            }
            అప్రమేయ: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((గణన rpc_cmd)cmd));
                తిరిగి;
            }
        }
    }
}

శూన్య ggml_backend_rpc_ప్రారంభం_server(స్థిరరాశి అక్షరం * endpoint, స్థిరరాశి అక్షరం * cache_dir,
                                   size_t n_దారాలు, size_t n_devices, ggml_backend_dev_t * devices) {
    ఒకవేళ (n_devices == 0 || devices == శూన్యసూచిక) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        తిరిగి;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_సంస్కరణ,
        RPC_PROTO_MINOR_సంస్కరణ,
        RPC_PROTO_PATCH_సంస్కరణ);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    కొరకు (size_t i = 0; i < n_devices; i++) {
        స్వయంచాలక dev = devices[i];
        size_t విడుదల, మొత్తం;
        ggml_backend_dev_జ్ఞాపకం(dev, &విడుదల, &మొత్తం);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 మొత్తం / 1024 / 1024, విడుదల / 1024 / 1024);
        స్వయంచాలక backend = ggml_backend_dev_init(dev, శూన్యసూచిక);
        ఒకవేళ (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            తిరిగి;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : శూన్యసూచిక;
        ఒకవేళ (reg) {
            స్వయంచాలక ggml_backend_సమితి_n_దారాలు_fn = (ggml_backend_సమితి_n_దారాలు_t) ggml_backend_reg_get_proc_చిరునామా(reg, "ggml_backend_set_n_threads");
            ఒకవేళ (ggml_backend_సమితి_n_దారాలు_fn) {
                ggml_backend_సమితి_n_దారాలు_fn(backend, n_దారాలు);
            }
        }
    }

    std::string host;
    పూర్ణాంకం పోర్టు;
    ఒకవేళ (!parse_endpoint(endpoint, host, పోర్టు)) {
        తిరిగి;
    }

#నిర్వచితమైతే GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#లేకపోతే
    LOG_INFO("  transport      : TCP\n");
#అయితే_ముగింపు // GGML_RPC_RDMA
    ఒకవేళ (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        తిరిగి;
    }
    స్వయంచాలక server_socket = socket_t::create_server(host.c_str(), పోర్టు);
    ఒకవేళ (server_socket == శూన్యసూచిక) {
        LOG_ERROR("Failed to create server socket\n");
        తిరిగి;
    }
    అయితే (నిజం) {
        స్వయంచాలక client_socket = server_socket->accept();
        ఒకవేళ (client_socket == శూన్యసూచిక) {
            LOG_ERROR("Failed to accept client connection\n");
            తిరిగి;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    కొరకు (స్వయంచాలక backend : backends) {
        ggml_backend_free(backend);
    }
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    తిరిగి ctx->name.c_str();
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    తిరిగి ctx->description.c_str();
}

స్థిర శూన్య ggml_backend_rpc_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_జ్ఞాపకం(ctx->endpoint.c_str(), ctx->device, విడుదల, మొత్తం);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    తిరిగి GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->caps = {
        /* .async                 = */ నిజం,
        /* .host_buffer           = */ అబద్ధం,
        /* .buffer_from_host_ptr  = */ అబద్ధం,
        /* .events                = */ నిజం,
        /* .mmap_support          = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    తిరిగి ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_వాడని(params);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_rpc_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    తిరిగి ggml_backend_rpc_బఫరు_type(ctx->endpoint.c_str(), ctx->device);

    GGML_వాడని(dev);
}

స్థిర తార్కిక ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    GGML_వాడని(dev);
    GGML_వాడని(op);
    //TODO: call the remote backend and cache the results
    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    ఒకవేళ (!buft || buft->iface.get_name != ggml_backend_rpc_బఫరు_type_name) {
        తిరిగి అబద్ధం;
    }
    ggml_backend_rpc_బఫరు_type_context * buft_ctx = (ggml_backend_rpc_బఫరు_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    తిరిగి buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

స్థిర ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    స్వయంచాలక dispatcher = get_dispatcher(ctx->endpoint);
    తిరిగి dispatcher->event_new(dev);
}

స్థిర శూన్య ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    స్వయంచాలక dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

స్థిర శూన్య ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    స్వయంచాలక dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ఖాళీ,
    /* .buffer_from_host_ptr = */ ఖాళీ,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ ఖాళీ,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

నిర్మాణం ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    తిరిగి ctx ? ctx->name.c_str() : "RPC";
}

స్థిర size_t ggml_backend_rpc_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    తిరిగి ctx ? ctx->devices.size() : 0;
}

స్థిర ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t సూచీ) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    ఒకవేళ (ctx == శూన్యసూచిక) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } లేకపోతే {
        GGML_ASSERT(సూచీ < ctx->devices.size());
        తిరిగి ctx->devices[సూచీ];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
నిర్మాణం ggml_backend_rpc_comm_context {
    నిర్మాణం rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

స్థిర శూన్య ggml_backend_rpc_comm_free(శూన్య * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ఒకవేళ (comm_ctx == శూన్యసూచిక) {
        తిరిగి;
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & rank : comm_ctx->ranks) {
        స్వయంచాలక request = std::make_shared<rpc_సందేశం_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, పరిమాణం(*request));
        rank.dispatcher->busy_spin_release();
    }
    తొలగించు comm_ctx;
}

స్థిర శూన్య * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ఒకవేళ (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    కొరకు (size_t i = 0; i < n_backends; i++) {
        ఒకవేళ (!ggml_backend_is_rpc(backends[i])) {
            తిరిగి శూన్యసూచిక;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        కొరకు (స్థిరరాశి స్వయంచాలక & rank : ranks) {
            ఒకవేళ (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                తిరిగి శూన్యసూచిక;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    పూర్ణాంకం port0;
    ఒకవేళ (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        తిరిగి శూన్యసూచిక;
    }
    స్థిరరాశి uint32_t comm_పోర్టు = (uint32_t) port0 + 1000;
    ఒకవేళ (host0.size() >= 64) {
        తిరిగి శూన్యసూచిక;
    }

    కొరకు (స్థిరరాశి స్వయంచాలక & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_సందేశం_comm_init_rsp> responses(n_backends);
    కొరకు (size_t i = 0; i < n_backends; i++) {
        స్వయంచాలక request = std::make_shared<rpc_సందేశం_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->పోర్టు   = comm_పోర్టు;
        ఒకవేళ (i > 0) {
            జ్ఞాపకం_నకలు(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, పరిమాణం(*request), &responses[i], పరిమాణం(responses[i]));
    }
    కొరకు (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    తార్కిక ok = నిజం;
    కొరకు (size_t i = 0; i < n_backends; i++) {
        ఒకవేళ (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = అబద్ధం;
        }
    }
    ఒకవేళ (!ok) {
        కొరకు (స్థిరరాశి స్వయంచాలక & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        తిరిగి శూన్యసూచిక;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    తిరిగి కొత్త ggml_backend_rpc_comm_context{std::move(ranks)};
}

స్థిర తార్కిక ggml_backend_rpc_comm_allreduce_tensor(శూన్య * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    ఒకవేళ (comm_ctx == శూన్యసూచిక) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి size_t n_ranks = comm_ctx->ranks.size();
    స్థిరరాశి int64_t ne = ggml_nelements(tensors[0]);
    ఒకవేళ (ne == 0) {
        తిరిగి నిజం;
    }
    కొరకు (size_t i = 0; i < n_ranks; i++) {
        ఒకవేళ (tensors[i] == శూన్యసూచిక || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->బఫరు == శూన్యసూచిక || !ggml_backend_బఫరు_is_rpc(tensors[i]->బఫరు)) {
            తిరిగి అబద్ధం;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        ఒకవేళ ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            తిరిగి అబద్ధం;
        }
    }
    కొరకు (size_t i = 0; i < n_ranks; i++) {
        స్వయంచాలక request = std::make_shared<rpc_సందేశం_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, పరిమాణం(*request));
    }
    తిరిగి నిజం;
}

స్థిర శూన్య * ggml_backend_rpc_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    ఒకవేళ (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        తిరిగి (శూన్య *)ggml_backend_rpc_కూడు_server;
    }
    ఒకవేళ (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        తిరిగి (శూన్య *)ggml_backend_rpc_ప్రారంభం_server;
    }
    ఒకవేళ (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        తిరిగి (శూన్య *)ggml_backend_rpc_comm_init;
    }
    ఒకవేళ (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        తిరిగి (శూన్య *)ggml_backend_rpc_comm_free;
    }
    ఒకవేళ (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        తిరిగి (శూన్య *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    తిరిగి ఖాళీ;

    GGML_వాడని(reg);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_చిరునామా,
};

ggml_backend_reg_t ggml_backend_rpc_reg(శూన్య) {
    స్థిర నిర్మాణం ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ ఖాళీ,
    };

    తిరిగి &ggml_backend_rpc_reg;
}

స్థిర uint32_t ggml_backend_rpc_get_device_లెక్క(స్థిరరాశి అక్షరం * endpoint) {
    స్వయంచాలక dispatcher = get_dispatcher(endpoint);
    rpc_సందేశం_device_లెక్క_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_లెక్క, శూన్యసూచిక, 0, &response, పరిమాణం(response));
    తిరిగి response.device_లెక్క;
}

స్థిర స్థిరరాశి ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_లెక్క,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_చిరునామా,
};

ggml_backend_reg_t ggml_backend_rpc_కూడు_server(స్థిరరాశి అక్షరం * endpoint) {
    స్థిర std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    స్థిర std::mutex mutex;
    స్థిర uint32_t dev_id = 0;
    std::lock_guard<std::mutex> తాళం(mutex);
    ఒకవేళ (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        తిరిగి reg_map[endpoint];
    }
    uint32_t dev_లెక్క = ggml_backend_rpc_get_device_లెక్క(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_లెక్క);
    ఒకవేళ (dev_లెక్క == 0) {
        తిరిగి శూన్యసూచిక;
    }
    ggml_backend_rpc_reg_context * ctx = కొత్త ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    కొరకు (uint32_t ind = 0; ind < dev_లెక్క; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = కొత్త ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = కొత్త ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = కొత్త ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    తిరిగి reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)

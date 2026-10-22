#సూచన once

#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>

నిర్మాణం socket_t;
రకనిర్వచనం std::shared_ptr<socket_t> socket_ptr;

స్థిర constexpr size_t MAX_CHUNK_కొలత = 1024ull * 1024ull * 1024ull; // 1 GiB
స్థిర constexpr size_t RPC_CONN_CAPS_కొలత = 24;

నిర్మాణం socket_t {
    ~socket_t();

    bool send_దత్తాంశం(స్థిరరాశి శూన్య * దత్తాంశం, size_t కొలత);
    bool recv_దత్తాంశం(శూన్య * దత్తాంశం, size_t కొలత);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    శూన్య get_caps(uint8_t * local_caps);
    శూన్య update_caps(స్థిరరాశి uint8_t * remote_caps);
    స్థిరరాశి అక్షరం * transport_name() స్థిరరాశి;

    స్థిర socket_ptr create_server(స్థిరరాశి అక్షరం * host, పూర్ణాంకం పోర్టు);
    స్థిర socket_ptr connect(స్థిరరాశి అక్షరం * host, పూర్ణాంకం పోర్టు);

అంతరంగ:
    నిర్మాణం impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
శూన్య rpc_transport_shutdown();

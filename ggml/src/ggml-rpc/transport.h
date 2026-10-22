#निर्देशः once

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>

संरचना socket_t;
प्रकारनिर्देश std::shared_ptr<socket_t> socket_ptr;

स्थिर constexpr size_t MAX_CHUNK_आकार = 1024ull * 1024ull * 1024ull; // 1 GiB
स्थिर constexpr size_t RPC_CONN_CAPS_आकार = 24;

संरचना socket_t {
    ~socket_t();

    bool send_दत्त(ध्रुव शून्य * दत्त, size_t आकार);
    bool recv_दत्त(शून्य * दत्त, size_t आकार);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    शून्य get_caps(uint8_t * local_caps);
    शून्य update_caps(ध्रुव uint8_t * remote_caps);
    ध्रुव वर्ण * transport_name() ध्रुव;

    स्थिर socket_ptr create_server(ध्रुव वर्ण * host, पूर्णाङ्क पत्तनम्);
    स्थिर socket_ptr connect(ध्रुव वर्ण * host, पूर्णाङ्क पत्तनम्);

गुप्त:
    संरचना impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
शून्य rpc_transport_shutdown();

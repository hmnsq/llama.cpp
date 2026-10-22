#pragma once

#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>

संरचना socket_t;
प्रकारव्याख्या std::shared_ptr<socket_t> socket_ptr;

स्थिर constexpr माप_प्रकार MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
स्थिर constexpr माप_प्रकार RPC_CONN_CAPS_SIZE = 24;

संरचना socket_t {
    ~socket_t();

    bool send_data(स्थिरांक शून्य * data, माप_प्रकार size);
    bool recv_data(शून्य * data, माप_प्रकार size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    शून्य get_caps(uint8_t * local_caps);
    शून्य update_caps(स्थिरांक uint8_t * remote_caps);
    स्थिरांक अक्षर * transport_name() स्थिरांक;

    स्थिर socket_ptr create_server(स्थिरांक अक्षर * host, पूर्णांक port);
    स्थिर socket_ptr connect(स्थिरांक अक्षर * host, पूर्णांक port);

खाजगी:
    संरचना impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
शून्य rpc_transport_shutdown();

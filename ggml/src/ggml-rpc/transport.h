#pragma once

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>

ಸಂರಚನಾ socket_t;
ಪ್ರಕಾರನಿರ್ದೇಶ std::shared_ptr<socket_t> socket_ptr;

ಸ್ಥಿರ constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
ಸ್ಥಿರ constexpr size_t RPC_CONN_CAPS_SIZE = 24;

ಸಂರಚನಾ socket_t {
    ~socket_t();

    bool send_data(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size);
    bool recv_data(ಶೂನ್ಯ * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    ಶೂನ್ಯ get_caps(uint8_t * local_caps);
    ಶೂನ್ಯ update_caps(ಧ್ರುವ uint8_t * remote_caps);
    ಧ್ರುವ ವರ್ಣ * transport_name() ಧ್ರುವ;

    ಸ್ಥಿರ socket_ptr create_server(ಧ್ರುವ ವರ್ಣ * host, ಪೂರ್ಣಾಙ್ಕ port);
    ಸ್ಥಿರ socket_ptr connect(ಧ್ರುವ ವರ್ಣ * host, ಪೂರ್ಣಾಙ್ಕ port);

ಗುಪ್ತ:
    ಸಂರಚನಾ impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
ಶೂನ್ಯ rpc_transport_shutdown();

#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>

ರಚನೆ socket_t;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::shared_ptr<socket_t> socket_ptr;

ಸ್ಥಿರ constexpr size_t MAX_CHUNK_ಗಾತ್ರ = 1024ull * 1024ull * 1024ull; // 1 GiB
ಸ್ಥಿರ constexpr size_t RPC_CONN_CAPS_ಗಾತ್ರ = 24;

ರಚನೆ socket_t {
    ~socket_t();

    bool send_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2);
    bool recv_ದತ್ತಾಂಶ(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    ಶೂನ್ಯ get_caps(uint8_t * local_caps);
    ಶೂನ್ಯ update_caps(ಸ್ಥಿರಾಂಕ uint8_t * remote_caps);
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * transport_name() ಸ್ಥಿರಾಂಕ;

    ಸ್ಥಿರ socket_ptr create_server(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * host, ಪೂರ್ಣಾಂಕ ಪೋರ್ಟ್);
    ಸ್ಥಿರ socket_ptr connect(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * host, ಪೂರ್ಣಾಂಕ ಪೋರ್ಟ್);

ಖಾಸಗಿ:
    ರಚನೆ impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
ಶೂನ್ಯ rpc_transport_shutdown();

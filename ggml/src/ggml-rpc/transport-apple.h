#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>

ರಚನೆ apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    ಸ್ಥಿರ std::unique_ptr<apple_rdma> probe(ಪೂರ್ಣಾಂಕ fd, ಸ್ಥಿರಾಂಕ uint8_t * ಗುರಿ_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ಸ್ಥಿರಾಂಕ uint8_t * caps);

    bool send(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2);
    bool recv(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಗಾತ್ರ_2);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ಸ್ಥಿರಾಂಕ;

ಖಾಸಗಿ:
    ರಚನೆ impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

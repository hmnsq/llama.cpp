#pragma once

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>

ಸಂರಚನಾ apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    ಸ್ಥಿರ std::unique_ptr<apple_rdma> probe(ಪೂರ್ಣಾಙ್ಕ fd, ಧ್ರುವ uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ಧ್ರುವ uint8_t * caps);

    bool send(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size);
    bool recv(ಶೂನ್ಯ * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ಧ್ರುವ;

ಗುಪ್ತ:
    ಸಂರಚನಾ impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

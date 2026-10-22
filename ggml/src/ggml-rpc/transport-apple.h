#pragma once

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <memory>

સંરચના apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    સ્થિર std::unique_ptr<apple_rdma> probe(પૂર્ણાંક fd, અચલ uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(અચલ uint8_t * caps);

    bool send(અચલ શૂન્ય * data, size_t size);
    bool recv(શૂન્ય * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() અચલ;

ખાનગી:
    સંરચના impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

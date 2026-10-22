#هدايت once

#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <memory>

بنية apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    ساكن std::unique_ptr<apple_rdma> probe(صحيح fd, ثابت uint8_t * هدف_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ثابت uint8_t * caps);

    bool send(ثابت فراغ * معلومات, size_t سائيز);
    bool recv(فراغ * معلومات, size_t سائيز);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ثابت;

خاص:
    بنية impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

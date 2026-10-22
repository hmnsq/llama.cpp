#ہدایت once

#شامل <cstddef>
#شامل <cstdint>
#شامل <memory>

ساخت apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    ساکن std::unique_ptr<apple_rdma> probe(صحیح fd, ثابت uint8_t * ہدف_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ثابت uint8_t * caps);

    bool send(ثابت خالی * ڈیٹا, size_t سائز);
    bool recv(خالی * ڈیٹا, size_t سائز);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ثابت;

نجی:
    ساخت impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

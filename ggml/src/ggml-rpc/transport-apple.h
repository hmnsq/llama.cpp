#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>

ਬਣਤਰ apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    ਸਥਿਰ std::unique_ptr<apple_rdma> probe(ਪੂਰਨ fd, ਅਚਲ uint8_t * ਨਿਸ਼ਾਨਾ_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ਅਚਲ uint8_t * caps);

    bool send(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ);
    bool recv(ਖਾਲੀ * ਅੰਕੜੇ, size_t ਮਾਪ);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ਅਚਲ;

ਨਿਜੀ:
    ਬਣਤਰ impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

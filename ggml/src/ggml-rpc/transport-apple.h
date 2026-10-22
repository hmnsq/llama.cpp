#pragma once

#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <memory>

সংরচনা apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    স্থির std::unique_ptr<apple_rdma> probe(পূর্ণাঙ্ক fd, ধ্রুৱ uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ধ্রুৱ uint8_t * caps);

    bool send(ধ্রুৱ শূন্য * data, size_t size);
    bool recv(শূন্য * data, size_t size);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ধ্রুৱ;

গুপ্ত:
    সংরচনা impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

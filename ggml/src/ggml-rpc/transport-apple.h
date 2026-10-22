#pragma once

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <memory>

কাঠামো apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    স্থির std::unique_ptr<apple_rdma> probe(পূর্ণসংখ্যা fd, ধ্রুবক uint8_t * target_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(ধ্রুবক uint8_t * caps);

    bool send(ধ্রুবক শূন্য * তথ্য, size_t আকার_2);
    bool recv(শূন্য * তথ্য, size_t আকার_2);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() ধ্রুবক;

ব্যক্তিগত:
    কাঠামো impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

#సూచన once

#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>

నిర్మాణం apple_rdma {
    // target_gid is 16 bytes in, caps is RPC_CONN_CAPS_SIZE bytes out.
    స్థిర std::unique_ptr<apple_rdma> probe(పూర్ణాంకం fd, స్థిరరాశి uint8_t * లక్ష్యం_gid, uint8_t * caps);
    ~apple_rdma();

    // Peer endpoint from its caps, which must be non-zero: this blocks on a
    // readiness handshake over fd that the peer only joins if it also has RDMA.
    bool activate(స్థిరరాశి uint8_t * caps);

    bool send(స్థిరరాశి శూన్య * దత్తాంశం, size_t కొలత);
    bool recv(శూన్య * దత్తాంశం, size_t కొలత);
    // Post the trailing partial frame; must be called at every message boundary.
    bool flush();
    // True once the connection has failed; the caller should drop the socket.
    bool broken() స్థిరరాశి;

అంతరంగ:
    నిర్మాణం impl;
    explicit apple_rdma(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

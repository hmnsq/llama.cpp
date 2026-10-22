#ਅਨ੍ਤਰ੍ਭਾਵ "transport-apple.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "transport.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "log.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <infiniband/verbs.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cerrno>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <dlfcn.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <poll.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/socket.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

// Apple RDMA-over-Thunderbolt (see Apple TN3205).
//
// Apple's RDMA is quite different from what's supported in Linux - deserving of its own transport implementation.
// see https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt for details
// at a high level the main differences are:
// UC(unreliable connection) on Apple vs RC(reliable connection) QP transport types on Linux (though in practice UC on Apple is still lossless)
// fixed 128KiB stride on Apple vs variable chunk size on Linux
// relying on Apple's hardware credit based flow control vs RNR NAKs + retries on Linux
//
// on Apple a SEND and its corresponding RECV must cover the same number of 4 KiB Thunderbolt frames,
// so every SEND posts a whole 128KiB stride over the wire, even when partially filled.
// (In testing 128KiB was the best performing among 32, 64, 128, 256)

//TODO: add mechanism similar to https://github.com/ggml-org/llama.cpp/pull/29440 to prevent idle CPU from spinning
ਸਥਿਰ constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
ਸਥਿਰ constexpr ਪੂਰਨ      RDMA_NBUF        = 16;          // ring depth (frames per direction)
ਸਥਿਰ constexpr size_t   RDMA_ਚੌਖਟਾ       = 4096;        // Thunderbolt frame (fixed on Apple)
ਸਥਿਰ constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
ਸਥਿਰ constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
ਸਥਿਰ constexpr size_t   RDMA_GID_ਮਾਪ    = 16;

static_assert(RDMA_STRIDE % RDMA_ਚੌਖਟਾ == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
ਸਥਿਰ constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_ਚੌਖਟਾ);
ਸਥਿਰ constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
ਸਥਿਰ constexpr uint64_t RDMA_WR_IDX_ਨਕਾਬ = 0xffff;      // buffer index in the low bits of wr_id
ਸਥਿਰ constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

ਬਣਤਰ rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
ਸਥਿਰ constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - ਆਕਾਰ(rdma_seg_hdr);

ਬਣਤਰ apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_ਮਾਪ];
};

static_assert(ਆਕਾਰ(apple_rdma_caps) == RPC_CONN_CAPS_ਮਾਪ, "apple_rdma_caps must match conn_caps size");

ਬਣਤਰ apple_rdma::impl {
    ਪੂਰਨ fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    ਬਣਤਰ ibv_context * ctx = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ibv_pd * pd = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ibv_cq * cq = ਖਾਲੀ_ਸੂਚਕ;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    ਬਣਤਰ ibv_qp * qp = ਖਾਲੀ_ਸੂਚਕ;

    uint8_t       * send_mem = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ibv_mr * send_mr  = ਖਾਲੀ_ਸੂਚਕ;
    uint8_t       * recv_mem = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ibv_mr * recv_mr  = ਖਾਲੀ_ਸੂਚਕ;

    ਪੂਰਨ      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    ਬਣਤਰ { ਪੂਰਨ buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    ਪੂਰਨ      inq_ਸਿਰ = 0;
    ਪੂਰਨ      inq_ਗਿਣਤੀ = 0;
    ਪੂਰਨ      pend_buf = -1;
    uint32_t pend_len = 0;
    ਤਾਰਕਿਕ     broken = ਝੂਠ;

    uint32_t     qpn = 0;
    uint8_t      ਪੋਰਟ = 0;
    ਪੂਰਨ          gid_idx = 0;
    ਗਿਣਤੀ ibv_mtu ਰਸਤਾ_mtu = IBV_MTU_1024;

    ਪੂਰਨ  ਤਰੱਕੀ();
    ਤਾਰਕਿਕ acquire_pending();
    ਤਾਰਕਿਕ post_pending();

    ਤਾਰਕਿਕ post_recv(ਪੂਰਨ i) {
        ਬਣਤਰ ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.ਲੰਮਾਈ = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        ਬਣਤਰ ibv_recv_wr wr = {}, * bad = ਖਾਲੀ_ਸੂਚਕ;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.ਅੰਕ_sge = 1;
        ਵਾਪਸ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ਤਾਰਕਿਕ post_send(ਪੂਰਨ i, size_t len) {
        ਬਣਤਰ ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.ਲੰਮਾਈ = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        ਬਣਤਰ ibv_send_wr wr = {}, * bad = ਖਾਲੀ_ਸੂਚਕ;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ਅੰਕ_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        ਵਾਪਸ ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = ਸੱਚ;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        ਜੇ (qp) ibv_destroy_qp(qp);
        ਜੇ (send_mr) ibv_dereg_mr(send_mr);
        ਜੇ (recv_mr) ibv_dereg_mr(recv_mr);
        ਮੁਕਤ(send_mem);
        ਮੁਕਤ(recv_mem);
        ਜੇ (cq)  ibv_destroy_cq(cq);
        ਜੇ (pd)  ibv_dealloc_pd(pd);
        ਜੇ (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = ਮੂਲ;

ਤਾਰਕਿਕ apple_rdma::broken() ਅਚਲ {
    ਵਾਪਸ pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
ਸਥਿਰ ਤਾਰਕਿਕ tcp_send_byte(ਪੂਰਨ fd, uint8_t b) {
    ssize_t n;
    ਕਰੋ { n = ::send(fd, &b, ਆਕਾਰ(b), 0); } ਜਦੋਂ (n < 0 && errno == EINTR);
    ਵਾਪਸ n == ਆਕਾਰ(b);
}

ਸਥਿਰ ਤਾਰਕਿਕ tcp_recv_byte(ਪੂਰਨ fd, uint8_t * b) {
    ssize_t n;
    ਕਰੋ { n = ::recv(fd, b, ਆਕਾਰ(*b), 0); } ਜਦੋਂ (n < 0 && errno == EINTR);
    ਵਾਪਸ n == (ssize_t)ਆਕਾਰ(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
ਸਥਿਰ ਪੂਰਨ rdma_ਮੇਲ_gid(ਬਣਤਰ ibv_context * ctx, uint8_t ਪੋਰਟ, ਪੂਰਨ gid_tbl_len,
                          ਅਚਲ uint8_t * ਨਿਸ਼ਾਨਾ, ਸੰਘ ibv_gid * ਬਾਹਰਲਾ) {
    ਲਈ (ਪੂਰਨ i = 0; i < gid_tbl_len; i++) {
        ਸੰਘ ibv_gid g;
        ਜੇ (ibv_query_gid(ctx, ਪੋਰਟ, i, &g) != 0) ਜਾਰੀ;
        ਜੇ (memcmp(g.raw, ਨਿਸ਼ਾਨਾ, RDMA_GID_ਮਾਪ) != 0) ਜਾਰੀ;
        ਜੇ (ਬਾਹਰਲਾ) *ਬਾਹਰਲਾ = g;
        ਵਾਪਸ i;
    }
    ਵਾਪਸ -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
ਸਥਿਰ uint8_t rdma_ਪਹਿਲਾ_ਸਰਗਰਮ_ਪੋਰਟ(ਬਣਤਰ ibv_context * ctx, ਬਣਤਰ ibv_ਪੋਰਟ_attr * ਬਾਹਰਲਾ) {
    ਬਣਤਰ ibv_device_attr da;
    ਜੇ (ibv_query_device(ctx, &da) != 0) ਵਾਪਸ 0;
    ਲਈ (uint8_t p = 1; p <= da.phys_ਪੋਰਟ_cnt; p++) {
        ਬਣਤਰ ibv_ਪੋਰਟ_attr pa;
        ਜੇ (ibv_query_ਪੋਰਟ(ctx, p, &pa) != 0) ਜਾਰੀ;
        ਜੇ (pa.state == IBV_ਪੋਰਟ_ਸਰਗਰਮ) { ਜੇ (ਬਾਹਰਲਾ) *ਬਾਹਰਲਾ = pa; ਵਾਪਸ p; }
    }
    ਵਾਪਸ 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
ਸਥਿਰ ਤਾਰਕਿਕ rdma_library_present() {
    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ present = [] {
        ਖਾਲੀ * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        ਜੇ (handle == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ਝੂਠ;
        }
        dlclose(handle);
        ਵਾਪਸ ਸੱਚ;
    }();
    ਵਾਪਸ present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(ਪੂਰਨ fd, ਅਚਲ uint8_t * ਨਿਸ਼ਾਨਾ_gid, uint8_t * caps) {
    ਜੇ (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਪੂਰਨ ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    ਜੇ (!devs) ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;

    ibv_context * ctx = ਖਾਲੀ_ਸੂਚਕ;
    uint8_t ਪੋਰਟ = 0;
    ਬਣਤਰ ibv_ਪੋਰਟ_attr pa = {};
    ਸੰਘ ibv_gid gid = {};
    ਪੂਰਨ gid_idx = -1;
    std::string matched;
    ਲਈ (ਪੂਰਨ d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        ਜੇ (!c) ਜਾਰੀ;
        ਬਣਤਰ ibv_ਪੋਰਟ_attr p = {};
        uint8_t pt = rdma_ਪਹਿਲਾ_ਸਰਗਰਮ_ਪੋਰਟ(c, &p);
        ਪੂਰਨ gi = pt ? rdma_ਮੇਲ_gid(c, pt, p.gid_tbl_len, ਨਿਸ਼ਾਨਾ_gid, &gid) : -1;
        ਜੇ (gi < 0) { ibv_close_device(c); ਜਾਰੀ; }
        ctx = c; ਪੋਰਟ = pt; pa = p; gid_idx = gi;
        ਅਚਲ ਅੱਖਰ * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        ਰੋਕੋ;
    }
    ibv_free_device_list(devs);
    ਜੇ (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    std::unique_ptr<impl> c(ਨਵਾਂ impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->ਪੋਰਟ = ਪੋਰਟ;
    c->gid_idx = gid_idx;
    c->ਰਸਤਾ_mtu = pa.ਸਰਗਰਮ_mtu;

    c->pd = ibv_alloc_pd(ctx);
    ਜੇ (!c->pd) ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ, 0);
    ਜੇ (!c->cq) ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    ਜੇ (!c->qp) ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_ਸੂਚਕਾਂਕ = 0;
        a.ਪੋਰਟ_ਅੰਕ = ਪੋਰਟ;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ਜੇ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_ਸੂਚਕਾਂਕ | IBV_QP_ਪੋਰਟ | IBV_QP_ACCESS_FLAGS) != 0) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    ਲੰਮਾ ਪੰਨਾ = sysconf(_SC_PAGESIZE);
    ਜੇ (ਪੰਨਾ <= 0) ਪੰਨਾ = 4096;
    ਅਚਲ size_t ring_ਬਾਈਟਾਂ = (size_t)RDMA_NBUF * RDMA_STRIDE;
    ਜੇ (posix_memalign((ਖਾਲੀ **)&c->send_mem, (size_t)ਪੰਨਾ, ring_ਬਾਈਟਾਂ) != 0) c->send_mem = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (posix_memalign((ਖਾਲੀ **)&c->recv_mem, (size_t)ਪੰਨਾ, ring_ਬਾਈਟਾਂ) != 0) c->recv_mem = ਖਾਲੀ_ਸੂਚਕ;
    ਜੇ (!c->send_mem || !c->recv_mem) ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    ਅਚਲ ਪੂਰਨ mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_ਬਾਈਟਾਂ, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_ਬਾਈਟਾਂ, mr_flags);
    ਜੇ (!c->send_mr || !c->recv_mr) ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_ਅੰਕ;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(rc.gid, gid.raw, RDMA_GID_ਮਾਪ);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(caps, &rc, ਆਕਾਰ(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), ਪੋਰਟ, gid_idx, c->qpn, (ਬਿਨਾਨਿਸ਼ਾਨੀ)pa.lid, 128 << c->ਰਸਤਾ_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    ਵਾਪਸ std::unique_ptr<apple_rdma>(ਨਵਾਂ apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
ਤਾਰਕਿਕ apple_rdma::activate(ਅਚਲ uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rc, caps, ਆਕਾਰ(rc));

    ਤਾਰਕਿਕ ok = ਸੱਚ;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.ਰਸਤਾ_mtu   = c->ਰਸਤਾ_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_ਅੰਕ = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.ਪੋਰਟ_ਅੰਕ      = c->ਪੋਰਟ;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_ਰਸਤਾ_ਬਿੱਟ = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_ਹੱਦਬੰਦੀ  = 1;
        a.ah_attr.grh.sgid_ਸੂਚਕਾਂਕ = (uint8_t)c->gid_idx;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_ਮਾਪ);
        ਜੇ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_ਰਸਤਾ_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = ਝੂਠ;
        }
    }
    ਜੇ (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        ਜੇ (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = ਝੂਠ;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    ਲਈ (ਪੂਰਨ i = 0; ok && i < RDMA_NBUF; i++) {
        ਜੇ (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = ਝੂਠ;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    ਜੇ (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ok || peer_ready != RDMA_SYNC_READY) {
        ਵਾਪਸ ਝੂਠ;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->ਰਸਤਾ_mtu, RDMA_NBUF);
    ਵਾਪਸ ਸੱਚ;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
ਪੂਰਨ apple_rdma::impl::ਤਰੱਕੀ() {
    ਬਣਤਰ ibv_wc wc[RDMA_NBUF * 2];
    ਪੂਰਨ n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    ਜੇ (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = ਸੱਚ; ਵਾਪਸ -1; }
    ਲਈ (ਪੂਰਨ j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        ਤਾਰਕਿਕ is_recv = (id & RDMA_RECV_WR) != 0;
        ਜੇ (wc[j].ਸਥਿਤੀ != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].ਸਥਿਤੀ);
            broken = ਸੱਚ;
            ਵਾਪਸ -1;
        }
        ਜੇ (is_recv) {
            ਪੂਰਨ b = (ਪੂਰਨ)(id & RDMA_WR_IDX_ਨਕਾਬ);
            ਅਚਲ rdma_seg_hdr * h = (ਅਚਲ rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            ਜੇ (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = ਸੱਚ; ਵਾਪਸ -1; }
            ਜੇ (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = ਸੱਚ; ਵਾਪਸ -1; }
            ਪੂਰਨ ਥਾਂ = (inq_ਸਿਰ + inq_ਗਿਣਤੀ) % RDMA_NBUF;
            inq[ਥਾਂ].buf  = b;
            inq[ਥਾਂ].off  = 0;
            inq[ਥਾਂ].len  = h->len;
            inq_ਗਿਣਤੀ++;
        } ਨਹੀਂਤਾਂ {
            send_busy[(ਪੂਰਨ)(id & RDMA_WR_IDX_ਨਕਾਬ)] = 0;
        }
    }
    ਵਾਪਸ n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
ਤਾਰਕਿਕ apple_rdma::impl::acquire_pending() {
    ਜੇ (pend_buf >= 0) ਵਾਪਸ ਸੱਚ;
    ਲਈ (;;) {
        ਜੇ (broken) ਵਾਪਸ ਝੂਠ;
        ਲਈ (ਪੂਰਨ k = 0; k < RDMA_NBUF; k++) ਜੇ (!send_busy[k]) { pend_buf = k; pend_len = 0; ਵਾਪਸ ਸੱਚ; }
        ਜੇ (ਤਰੱਕੀ() < 0) ਵਾਪਸ ਝੂਠ;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
ਤਾਰਕਿਕ apple_rdma::impl::post_pending() {
    ਜੇ (pend_buf < 0) ਵਾਪਸ ਸੱਚ;
    ਪੂਰਨ i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    ਜੇ (!post_send(i, RDMA_STRIDE)) { broken = ਸੱਚ; ਵਾਪਸ ਝੂਠ; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    ਵਾਪਸ ਸੱਚ;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
ਤਾਰਕਿਕ apple_rdma::send(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    impl * c = pimpl.get();
    ਅਚਲ uint8_t * p = (ਅਚਲ uint8_t *)ਅੰਕੜੇ;
    ਜਦੋਂ (size > 0) {
        ਜੇ (c->broken) ਵਾਪਸ ਝੂਠ;
        ਜੇ (!c->acquire_pending()) ਵਾਪਸ ਝੂਠ;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(sb + ਆਕਾਰ(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        ਜੇ (c->pend_len == RDMA_PAYLOAD) { ਜੇ (!c->post_pending()) ਵਾਪਸ ਝੂਠ; }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ apple_rdma::recv(ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)ਅੰਕੜੇ;
    ਜੇ (!c->post_pending()) ਵਾਪਸ ਝੂਠ;   // turnaround: flush the coalesced request
    ਬਿਨਾਨਿਸ਼ਾਨੀ idle = 0;
    ਜਦੋਂ (size > 0) {
        ਜੇ (c->inq_ਗਿਣਤੀ == 0) {
            ਜੇ (c->broken) ਵਾਪਸ ਝੂਠ;
            ਪੂਰਨ n = c->ਤਰੱਕੀ();
            ਜੇ (n < 0) ਵਾਪਸ ਝੂਠ;
            ਜੇ (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                ਜੇ ((++idle & 0xFFFFF) == 0) {
                    ਬਣਤਰ pollfd pfd = { c->fd, POLLIN, 0 };
                    ਜੇ (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                idle = 0;
            }
            ਜਾਰੀ;
        }
        idle = 0;
        ਪੂਰਨ ਥਾਂ = c->inq_ਸਿਰ;
        ਪੂਰਨ b = c->inq[ਥਾਂ].buf;
        uint32_t avail = c->inq[ਥਾਂ].len - c->inq[ਥਾਂ].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(p, c->recv_mem + (size_t)b * RDMA_STRIDE + ਆਕਾਰ(rdma_seg_hdr) + c->inq[ਥਾਂ].off, take);
        p += take;
        size -= take;
        c->inq[ਥਾਂ].off += take;
        ਜੇ (c->inq[ਥਾਂ].off == c->inq[ਥਾਂ].len) {
            ਜੇ (!c->post_recv(b)) { c->broken = ਸੱਚ; ਵਾਪਸ ਝੂਠ; }
            c->inq_ਸਿਰ = (c->inq_ਸਿਰ + 1) % RDMA_NBUF;
            c->inq_ਗਿਣਤੀ--;
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ apple_rdma::flush() {
    ਵਾਪਸ pimpl->post_pending();
}

#ಅನ್ತರ್ಭಾವ "transport-apple.h"
#ಅನ್ತರ್ಭಾವ "transport.h"
#ಅನ್ತರ್ಭಾವ "log.h"

#ಅನ್ತರ್ಭಾವ <infiniband/verbs.h>

#ಅನ್ತರ್ಭಾವ <cerrno>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <dlfcn.h>
#ಅನ್ತರ್ಭಾವ <poll.h>
#ಅನ್ತರ್ಭಾವ <sys/socket.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

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
ಸ್ಥಿರ constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ      RDMA_NBUF        = 16;          // ring depth (frames per direction)
ಸ್ಥಿರ constexpr size_t   RDMA_ಚೌಕಟ್ಟು       = 4096;        // Thunderbolt frame (fixed on Apple)
ಸ್ಥಿರ constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
ಸ್ಥಿರ constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
ಸ್ಥಿರ constexpr size_t   RDMA_GID_ಗಾತ್ರ    = 16;

static_assert(RDMA_STRIDE % RDMA_ಚೌಕಟ್ಟು == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
ಸ್ಥಿರ constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_ಚೌಕಟ್ಟು);
ಸ್ಥಿರ constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
ಸ್ಥಿರ constexpr uint64_t RDMA_WR_IDX_ಮುಖವಾಡ = 0xffff;      // buffer index in the low bits of wr_id
ಸ್ಥಿರ constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

ರಚನೆ rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
ಸ್ಥಿರ constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - ಗಾತ್ರ(rdma_seg_hdr);

ರಚನೆ apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_ಗಾತ್ರ];
};

static_assert(ಗಾತ್ರ(apple_rdma_caps) == RPC_CONN_CAPS_ಗಾತ್ರ, "apple_rdma_caps must match conn_caps size");

ರಚನೆ apple_rdma::impl {
    ಪೂರ್ಣಾಂಕ fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    ರಚನೆ ibv_context * ctx = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ibv_pd * pd = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ibv_cq * cq = ಶೂನ್ಯಸೂಚಕ;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    ರಚನೆ ibv_qp * qp = ಶೂನ್ಯಸೂಚಕ;

    uint8_t       * send_mem = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ibv_mr * send_mr  = ಶೂನ್ಯಸೂಚಕ;
    uint8_t       * recv_mem = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ibv_mr * recv_mr  = ಶೂನ್ಯಸೂಚಕ;

    ಪೂರ್ಣಾಂಕ      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    ರಚನೆ { ಪೂರ್ಣಾಂಕ buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    ಪೂರ್ಣಾಂಕ      inq_ತಲೆ = 0;
    ಪೂರ್ಣಾಂಕ      inq_ಎಣಿಕೆ = 0;
    ಪೂರ್ಣಾಂಕ      pend_buf = -1;
    uint32_t pend_len = 0;
    ತಾರ್ಕಿಕ     broken = ಸುಳ್ಳು;

    uint32_t     qpn = 0;
    uint8_t      ಪೋರ್ಟ್ = 0;
    ಪೂರ್ಣಾಂಕ          gid_idx = 0;
    ಎಣಿಕೆ ibv_mtu ಮಾರ್ಗ_mtu = IBV_MTU_1024;

    ಪೂರ್ಣಾಂಕ  ಪ್ರಗತಿ();
    ತಾರ್ಕಿಕ acquire_pending();
    ತಾರ್ಕಿಕ post_pending();

    ತಾರ್ಕಿಕ post_recv(ಪೂರ್ಣಾಂಕ i) {
        ರಚನೆ ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.ಉದ್ದಳತೆ = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        ರಚನೆ ibv_recv_wr wr = {}, * bad = ಶೂನ್ಯಸೂಚಕ;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.ಅಂಕೆ_sge = 1;
        ಹಿಂತಿರುಗಿ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ತಾರ್ಕಿಕ post_send(ಪೂರ್ಣಾಂಕ i, size_t len) {
        ರಚನೆ ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.ಉದ್ದಳತೆ = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        ರಚನೆ ibv_send_wr wr = {}, * bad = ಶೂನ್ಯಸೂಚಕ;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ಅಂಕೆ_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        ಹಿಂತಿರುಗಿ ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = ನಿಜ;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        ಆದರೆ (qp) ibv_destroy_qp(qp);
        ಆದರೆ (send_mr) ibv_dereg_mr(send_mr);
        ಆದರೆ (recv_mr) ibv_dereg_mr(recv_mr);
        ಬಿಡುಗಡೆ(send_mem);
        ಬಿಡುಗಡೆ(recv_mem);
        ಆದರೆ (cq)  ibv_destroy_cq(cq);
        ಆದರೆ (pd)  ibv_dealloc_pd(pd);
        ಆದರೆ (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = ಪೂರ್ವನಿಯೋಜಿತ;

ತಾರ್ಕಿಕ apple_rdma::broken() ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
ಸ್ಥಿರ ತಾರ್ಕಿಕ tcp_send_byte(ಪೂರ್ಣಾಂಕ fd, uint8_t b) {
    ssize_t n;
    ಮಾಡು { n = ::send(fd, &b, ಗಾತ್ರ(b), 0); } ಆಗ (n < 0 && errno == EINTR);
    ಹಿಂತಿರುಗಿ n == ಗಾತ್ರ(b);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ tcp_recv_byte(ಪೂರ್ಣಾಂಕ fd, uint8_t * b) {
    ssize_t n;
    ಮಾಡು { n = ::recv(fd, b, ಗಾತ್ರ(*b), 0); } ಆಗ (n < 0 && errno == EINTR);
    ಹಿಂತಿರುಗಿ n == (ssize_t)ಗಾತ್ರ(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ rdma_ಹೊಂದಿಕೆ_gid(ರಚನೆ ibv_context * ctx, uint8_t ಪೋರ್ಟ್, ಪೂರ್ಣಾಂಕ gid_tbl_len,
                          ಸ್ಥಿರಾಂಕ uint8_t * ಗುರಿ, ಒಕ್ಕೂಟ ibv_gid * ಹೊರಹೋಗು) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < gid_tbl_len; i++) {
        ಒಕ್ಕೂಟ ibv_gid g;
        ಆದರೆ (ibv_query_gid(ctx, ಪೋರ್ಟ್, i, &g) != 0) ಮುಂದುವರಿ;
        ಆದರೆ (memcmp(g.raw, ಗುರಿ, RDMA_GID_ಗಾತ್ರ) != 0) ಮುಂದುವರಿ;
        ಆದರೆ (ಹೊರಹೋಗು) *ಹೊರಹೋಗು = g;
        ಹಿಂತಿರುಗಿ i;
    }
    ಹಿಂತಿರುಗಿ -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
ಸ್ಥಿರ uint8_t rdma_ಮೊದಲ_ಸಕ್ರಿಯ_ಪೋರ್ಟ್(ರಚನೆ ibv_context * ctx, ರಚನೆ ibv_ಪೋರ್ಟ್_attr * ಹೊರಹೋಗು) {
    ರಚನೆ ibv_device_attr da;
    ಆದರೆ (ibv_query_device(ctx, &da) != 0) ಹಿಂತಿರುಗಿ 0;
    ಗಾಗಿ (uint8_t p = 1; p <= da.phys_ಪೋರ್ಟ್_cnt; p++) {
        ರಚನೆ ibv_ಪೋರ್ಟ್_attr pa;
        ಆದರೆ (ibv_query_ಪೋರ್ಟ್(ctx, p, &pa) != 0) ಮುಂದುವರಿ;
        ಆದರೆ (pa.state == IBV_ಪೋರ್ಟ್_ಸಕ್ರಿಯ) { ಆದರೆ (ಹೊರಹೋಗು) *ಹೊರಹೋಗು = pa; ಹಿಂತಿರುಗಿ p; }
    }
    ಹಿಂತಿರುಗಿ 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
ಸ್ಥಿರ ತಾರ್ಕಿಕ rdma_library_present() {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ present = [] {
        ಶೂನ್ಯ * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        ಆದರೆ (handle == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        dlclose(handle);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }();
    ಹಿಂತಿರುಗಿ present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(ಪೂರ್ಣಾಂಕ fd, ಸ್ಥಿರಾಂಕ uint8_t * ಗುರಿ_gid, uint8_t * caps) {
    ಆದರೆ (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಪೂರ್ಣಾಂಕ ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    ಆದರೆ (!devs) ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;

    ibv_context * ctx = ಶೂನ್ಯಸೂಚಕ;
    uint8_t ಪೋರ್ಟ್ = 0;
    ರಚನೆ ibv_ಪೋರ್ಟ್_attr pa = {};
    ಒಕ್ಕೂಟ ibv_gid gid = {};
    ಪೂರ್ಣಾಂಕ gid_idx = -1;
    std::string matched;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        ಆದರೆ (!c) ಮುಂದುವರಿ;
        ರಚನೆ ibv_ಪೋರ್ಟ್_attr p = {};
        uint8_t pt = rdma_ಮೊದಲ_ಸಕ್ರಿಯ_ಪೋರ್ಟ್(c, &p);
        ಪೂರ್ಣಾಂಕ gi = pt ? rdma_ಹೊಂದಿಕೆ_gid(c, pt, p.gid_tbl_len, ಗುರಿ_gid, &gid) : -1;
        ಆದರೆ (gi < 0) { ibv_close_device(c); ಮುಂದುವರಿ; }
        ctx = c; ಪೋರ್ಟ್ = pt; pa = p; gid_idx = gi;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        ನಿಲ್ಲಿಸು;
    }
    ibv_free_device_list(devs);
    ಆದರೆ (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    std::unique_ptr<impl> c(ಹೊಸ impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->ಪೋರ್ಟ್ = ಪೋರ್ಟ್;
    c->gid_idx = gid_idx;
    c->ಮಾರ್ಗ_mtu = pa.ಸಕ್ರಿಯ_mtu;

    c->pd = ibv_alloc_pd(ctx);
    ಆದರೆ (!c->pd) ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, 0);
    ಆದರೆ (!c->cq) ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    ಆದರೆ (!c->qp) ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_ಸೂಚ್ಯಂಕ = 0;
        a.ಪೋರ್ಟ್_ಅಂಕೆ = ಪೋರ್ಟ್;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ಆದರೆ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_ಸೂಚ್ಯಂಕ | IBV_QP_ಪೋರ್ಟ್ | IBV_QP_ACCESS_FLAGS) != 0) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }

    ದೀರ್ಘ ಪುಟ = sysconf(_SC_PAGESIZE);
    ಆದರೆ (ಪುಟ <= 0) ಪುಟ = 4096;
    ಸ್ಥಿರಾಂಕ size_t ring_ಬೈಟುಗಳು = (size_t)RDMA_NBUF * RDMA_STRIDE;
    ಆದರೆ (posix_memalign((ಶೂನ್ಯ **)&c->send_mem, (size_t)ಪುಟ, ring_ಬೈಟುಗಳು) != 0) c->send_mem = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (posix_memalign((ಶೂನ್ಯ **)&c->recv_mem, (size_t)ಪುಟ, ring_ಬೈಟುಗಳು) != 0) c->recv_mem = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (!c->send_mem || !c->recv_mem) ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_ಬೈಟುಗಳು, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_ಬೈಟುಗಳು, mr_flags);
    ಆದರೆ (!c->send_mr || !c->recv_mr) ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_ಅಂಕೆ;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    ಸ್ಮರಣೆ_ನಕಲು(rc.gid, gid.raw, RDMA_GID_ಗಾತ್ರ);
    ಸ್ಮರಣೆ_ನಕಲು(caps, &rc, ಗಾತ್ರ(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), ಪೋರ್ಟ್, gid_idx, c->qpn, (ಚಿಹ್ನೆರಹಿತ)pa.lid, 128 << c->ಮಾರ್ಗ_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    ಹಿಂತಿರುಗಿ std::unique_ptr<apple_rdma>(ಹೊಸ apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
ತಾರ್ಕಿಕ apple_rdma::activate(ಸ್ಥಿರಾಂಕ uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    ಸ್ಮರಣೆ_ನಕಲು(&rc, caps, ಗಾತ್ರ(rc));

    ತಾರ್ಕಿಕ ok = ನಿಜ;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.ಮಾರ್ಗ_mtu   = c->ಮಾರ್ಗ_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_ಅಂಕೆ = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.ಪೋರ್ಟ್_ಅಂಕೆ      = c->ಪೋರ್ಟ್;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_ಮಾರ್ಗ_ಬಿಟ್ಟುಗಳು = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_ಎಲ್ಲೆ  = 1;
        a.ah_attr.grh.sgid_ಸೂಚ್ಯಂಕ = (uint8_t)c->gid_idx;
        ಸ್ಮರಣೆ_ನಕಲು(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_ಗಾತ್ರ);
        ಆದರೆ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_ಮಾರ್ಗ_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = ಸುಳ್ಳು;
        }
    }
    ಆದರೆ (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        ಆದರೆ (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = ಸುಳ್ಳು;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; ok && i < RDMA_NBUF; i++) {
        ಆದರೆ (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = ಸುಳ್ಳು;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    ಆದರೆ (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ok || peer_ready != RDMA_SYNC_READY) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->ಮಾರ್ಗ_mtu, RDMA_NBUF);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
ಪೂರ್ಣಾಂಕ apple_rdma::impl::ಪ್ರಗತಿ() {
    ರಚನೆ ibv_wc wc[RDMA_NBUF * 2];
    ಪೂರ್ಣಾಂಕ n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    ಆದರೆ (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = ನಿಜ; ಹಿಂತಿರುಗಿ -1; }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        ತಾರ್ಕಿಕ is_recv = (id & RDMA_RECV_WR) != 0;
        ಆದರೆ (wc[j].ಸ್ಥಿತಿ_2 != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].ಸ್ಥಿತಿ_2);
            broken = ನಿಜ;
            ಹಿಂತಿರುಗಿ -1;
        }
        ಆದರೆ (is_recv) {
            ಪೂರ್ಣಾಂಕ b = (ಪೂರ್ಣಾಂಕ)(id & RDMA_WR_IDX_ಮುಖವಾಡ);
            ಸ್ಥಿರಾಂಕ rdma_seg_hdr * h = (ಸ್ಥಿರಾಂಕ rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            ಆದರೆ (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = ನಿಜ; ಹಿಂತಿರುಗಿ -1; }
            ಆದರೆ (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = ನಿಜ; ಹಿಂತಿರುಗಿ -1; }
            ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = (inq_ತಲೆ + inq_ಎಣಿಕೆ) % RDMA_NBUF;
            inq[ಸ್ಥಾನ].buf  = b;
            inq[ಸ್ಥಾನ].off  = 0;
            inq[ಸ್ಥಾನ].len  = h->len;
            inq_ಎಣಿಕೆ++;
        } ಇಲ್ಲದಿದ್ದರೆ {
            send_busy[(ಪೂರ್ಣಾಂಕ)(id & RDMA_WR_IDX_ಮುಖವಾಡ)] = 0;
        }
    }
    ಹಿಂತಿರುಗಿ n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
ತಾರ್ಕಿಕ apple_rdma::impl::acquire_pending() {
    ಆದರೆ (pend_buf >= 0) ಹಿಂತಿರುಗಿ ನಿಜ;
    ಗಾಗಿ (;;) {
        ಆದರೆ (broken) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < RDMA_NBUF; k++) ಆದರೆ (!send_busy[k]) { pend_buf = k; pend_len = 0; ಹಿಂತಿರುಗಿ ನಿಜ; }
        ಆದರೆ (ಪ್ರಗತಿ() < 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
ತಾರ್ಕಿಕ apple_rdma::impl::post_pending() {
    ಆದರೆ (pend_buf < 0) ಹಿಂತಿರುಗಿ ನಿಜ;
    ಪೂರ್ಣಾಂಕ i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    ಆದರೆ (!post_send(i, RDMA_STRIDE)) { broken = ನಿಜ; ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
ತಾರ್ಕಿಕ apple_rdma::send(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    impl * c = pimpl.get();
    ಸ್ಥಿರಾಂಕ uint8_t * p = (ಸ್ಥಿರಾಂಕ uint8_t *)ದತ್ತಾಂಶ;
    ಆಗ (size > 0) {
        ಆದರೆ (c->broken) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (!c->acquire_pending()) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        ಸ್ಮರಣೆ_ನಕಲು(sb + ಗಾತ್ರ(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        ಆದರೆ (c->pend_len == RDMA_PAYLOAD) { ಆದರೆ (!c->post_pending()) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ apple_rdma::recv(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)ದತ್ತಾಂಶ;
    ಆದರೆ (!c->post_pending()) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;   // turnaround: flush the coalesced request
    ಚಿಹ್ನೆರಹಿತ idle = 0;
    ಆಗ (size > 0) {
        ಆದರೆ (c->inq_ಎಣಿಕೆ == 0) {
            ಆದರೆ (c->broken) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಪೂರ್ಣಾಂಕ n = c->ಪ್ರಗತಿ();
            ಆದರೆ (n < 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಆದರೆ (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                ಆದರೆ ((++idle & 0xFFFFF) == 0) {
                    ರಚನೆ pollfd pfd = { c->fd, POLLIN, 0 };
                    ಆದರೆ (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                idle = 0;
            }
            ಮುಂದುವರಿ;
        }
        idle = 0;
        ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = c->inq_ತಲೆ;
        ಪೂರ್ಣಾಂಕ b = c->inq[ಸ್ಥಾನ].buf;
        uint32_t avail = c->inq[ಸ್ಥಾನ].len - c->inq[ಸ್ಥಾನ].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        ಸ್ಮರಣೆ_ನಕಲು(p, c->recv_mem + (size_t)b * RDMA_STRIDE + ಗಾತ್ರ(rdma_seg_hdr) + c->inq[ಸ್ಥಾನ].off, take);
        p += take;
        size -= take;
        c->inq[ಸ್ಥಾನ].off += take;
        ಆದರೆ (c->inq[ಸ್ಥಾನ].off == c->inq[ಸ್ಥಾನ].len) {
            ಆದರೆ (!c->post_recv(b)) { c->broken = ನಿಜ; ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
            c->inq_ತಲೆ = (c->inq_ತಲೆ + 1) % RDMA_NBUF;
            c->inq_ಎಣಿಕೆ--;
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ apple_rdma::flush() {
    ಹಿಂತಿರುಗಿ pimpl->post_pending();
}

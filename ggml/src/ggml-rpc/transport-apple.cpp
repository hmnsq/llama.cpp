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
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ      RDMA_NBUF        = 16;          // ring depth (frames per direction)
ಸ್ಥಿರ constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
ಸ್ಥಿರ constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
ಸ್ಥಿರ constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
ಸ್ಥಿರ constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
ಸ್ಥಿರ constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
ಸ್ಥಿರ constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
ಸ್ಥಿರ constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
ಸ್ಥಿರ constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

ಸಂರಚನಾ rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
ಸ್ಥಿರ constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - ಪರಿಮಾಣ(rdma_seg_hdr);

ಸಂರಚನಾ apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(ಪರಿಮಾಣ(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

ಸಂರಚನಾ apple_rdma::impl {
    ಪೂರ್ಣಾಙ್ಕ fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    ಸಂರಚನಾ ibv_context * ctx = nullptr;
    ಸಂರಚನಾ ibv_pd * pd = nullptr;
    ಸಂರಚನಾ ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    ಸಂರಚನಾ ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    ಸಂರಚನಾ ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    ಸಂರಚನಾ ibv_mr * recv_mr  = nullptr;

    ಪೂರ್ಣಾಙ್ಕ      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    ಸಂರಚನಾ { ಪೂರ್ಣಾಙ್ಕ buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    ಪೂರ್ಣಾಙ್ಕ      inq_head = 0;
    ಪೂರ್ಣಾಙ್ಕ      inq_ಎಣಿಕೆ = 0;
    ಪೂರ್ಣಾಙ್ಕ      pend_buf = -1;
    uint32_t pend_len = 0;
    ತಾರ್ಕಿಕ     broken = ಅಸತ್ಯ;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    ಪೂರ್ಣಾಙ್ಕ          gid_idx = 0;
    ಗಣನಾ ibv_mtu path_mtu = IBV_MTU_1024;

    ಪೂರ್ಣಾಙ್ಕ  progress();
    ತಾರ್ಕಿಕ acquire_pending();
    ತಾರ್ಕಿಕ post_pending();

    ತಾರ್ಕಿಕ post_recv(ಪೂರ್ಣಾಙ್ಕ i) {
        ಸಂರಚನಾ ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        ಸಂರಚನಾ ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.ಅಂಕೆ_sge = 1;
        ಪ್ರತ್ಯಾಗಮ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ತಾರ್ಕಿಕ post_send(ಪೂರ್ಣಾಙ್ಕ i, size_t len) {
        ಸಂರಚನಾ ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        ಸಂರಚನಾ ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ಅಂಕೆ_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        ಪ್ರತ್ಯಾಗಮ ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = ಸತ್ಯ;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        ಯದಿ (qp) ibv_destroy_qp(qp);
        ಯದಿ (send_mr) ibv_dereg_mr(send_mr);
        ಯದಿ (recv_mr) ibv_dereg_mr(recv_mr);
        ಮುಕ್ತ(send_mem);
        ಮುಕ್ತ(recv_mem);
        ಯದಿ (cq)  ibv_destroy_cq(cq);
        ಯದಿ (pd)  ibv_dealloc_pd(pd);
        ಯದಿ (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = ಸ್ವಭಾವ;

ತಾರ್ಕಿಕ apple_rdma::broken() ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
ಸ್ಥಿರ ತಾರ್ಕಿಕ tcp_send_byte(ಪೂರ್ಣಾಙ್ಕ fd, uint8_t b) {
    ssize_t n;
    ಕುರು { n = ::send(fd, &b, ಪರಿಮಾಣ(b), 0); } ಯಾವತ್ (n < 0 && errno == EINTR);
    ಪ್ರತ್ಯಾಗಮ n == ಪರಿಮಾಣ(b);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ tcp_recv_byte(ಪೂರ್ಣಾಙ್ಕ fd, uint8_t * b) {
    ssize_t n;
    ಕುರು { n = ::recv(fd, b, ಪರಿಮಾಣ(*b), 0); } ಯಾವತ್ (n < 0 && errno == EINTR);
    ಪ್ರತ್ಯಾಗಮ n == (ssize_t)ಪರಿಮಾಣ(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ rdma_match_gid(ಸಂರಚನಾ ibv_context * ctx, uint8_t port, ಪೂರ್ಣಾಙ್ಕ gid_tbl_len,
                          ಧ್ರುವ uint8_t * target, ಸಙ್ಘ ibv_gid * out) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < gid_tbl_len; i++) {
        ಸಙ್ಘ ibv_gid g;
        ಯದಿ (ibv_query_gid(ctx, port, i, &g) != 0) ಅನುವರ್ತ;
        ಯದಿ (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) ಅನುವರ್ತ;
        ಯದಿ (out) *out = g;
        ಪ್ರತ್ಯಾಗಮ i;
    }
    ಪ್ರತ್ಯಾಗಮ -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
ಸ್ಥಿರ uint8_t rdma_first_active_port(ಸಂರಚನಾ ibv_context * ctx, ಸಂರಚನಾ ibv_port_attr * out) {
    ಸಂರಚನಾ ibv_device_attr da;
    ಯದಿ (ibv_query_device(ctx, &da) != 0) ಪ್ರತ್ಯಾಗಮ 0;
    ಪ್ರತಿ (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        ಸಂರಚನಾ ibv_port_attr pa;
        ಯದಿ (ibv_query_port(ctx, p, &pa) != 0) ಅನುವರ್ತ;
        ಯದಿ (pa.state == IBV_PORT_ACTIVE) { ಯದಿ (out) *out = pa; ಪ್ರತ್ಯಾಗಮ p; }
    }
    ಪ್ರತ್ಯಾಗಮ 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
ಸ್ಥಿರ ತಾರ್ಕಿಕ rdma_library_present() {
    ಸ್ಥಿರ ಧ್ರುವ ತಾರ್ಕಿಕ present = [] {
        ಶೂನ್ಯ * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        ಯದಿ (handle == nullptr) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        dlclose(handle);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }();
    ಪ್ರತ್ಯಾಗಮ present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(ಪೂರ್ಣಾಙ್ಕ fd, ಧ್ರುವ uint8_t * target_gid, uint8_t * caps) {
    ಯದಿ (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪೂರ್ಣಾಙ್ಕ ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    ಯದಿ (!devs) ಪ್ರತ್ಯಾಗಮ nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    ಸಂರಚನಾ ibv_port_attr pa = {};
    ಸಙ್ಘ ibv_gid gid = {};
    ಪೂರ್ಣಾಙ್ಕ gid_idx = -1;
    std::string matched;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        ಯದಿ (!c) ಅನುವರ್ತ;
        ಸಂರಚನಾ ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        ಪೂರ್ಣಾಙ್ಕ gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        ಯದಿ (gi < 0) { ibv_close_device(c); ಅನುವರ್ತ; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        ಧ್ರುವ ವರ್ಣ * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        ಭಙ್ಗ;
    }
    ibv_free_device_list(devs);
    ಯದಿ (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    std::unique_ptr<impl> c(ನೂತನ impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    ಯದಿ (!c->pd) ಪ್ರತ್ಯಾಗಮ nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    ಯದಿ (!c->cq) ಪ್ರತ್ಯಾಗಮ nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    ಯದಿ (!c->qp) ಪ್ರತ್ಯಾಗಮ nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_ಅಂಕೆ = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ಯದಿ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    ಯದಿ (page <= 0) page = 4096;
    ಧ್ರುವ size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    ಯದಿ (posix_memalign((ಶೂನ್ಯ **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    ಯದಿ (posix_memalign((ಶೂನ್ಯ **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    ಯದಿ (!c->send_mem || !c->recv_mem) ಪ್ರತ್ಯಾಗಮ nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    ಯದಿ (!c->send_mr || !c->recv_mr) ಪ್ರತ್ಯಾಗಮ nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_ಅಂಕೆ;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, ಪರಿಮಾಣ(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    ಪ್ರತ್ಯಾಗಮ std::unique_ptr<apple_rdma>(ನೂತನ apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
ತಾರ್ಕಿಕ apple_rdma::activate(ಧ್ರುವ uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, ಪರಿಮಾಣ(rc));

    ತಾರ್ಕಿಕ ok = ಸತ್ಯ;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_ಅಂಕೆ = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_ಅಂಕೆ      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        ಯದಿ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = ಅಸತ್ಯ;
        }
    }
    ಯದಿ (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        ಯದಿ (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = ಅಸತ್ಯ;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; ok && i < RDMA_NBUF; i++) {
        ಯದಿ (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = ಅಸತ್ಯ;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    ಯದಿ (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (!ok || peer_ready != RDMA_SYNC_READY) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
ಪೂರ್ಣಾಙ್ಕ apple_rdma::impl::progress() {
    ಸಂರಚನಾ ibv_wc wc[RDMA_NBUF * 2];
    ಪೂರ್ಣಾಙ್ಕ n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    ಯದಿ (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = ಸತ್ಯ; ಪ್ರತ್ಯಾಗಮ -1; }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        ತಾರ್ಕಿಕ is_recv = (id & RDMA_RECV_WR) != 0;
        ಯದಿ (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = ಸತ್ಯ;
            ಪ್ರತ್ಯಾಗಮ -1;
        }
        ಯದಿ (is_recv) {
            ಪೂರ್ಣಾಙ್ಕ b = (ಪೂರ್ಣಾಙ್ಕ)(id & RDMA_WR_IDX_MASK);
            ಧ್ರುವ rdma_seg_hdr * h = (ಧ್ರುವ rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            ಯದಿ (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = ಸತ್ಯ; ಪ್ರತ್ಯಾಗಮ -1; }
            ಯದಿ (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = ಸತ್ಯ; ಪ್ರತ್ಯಾಗಮ -1; }
            ಪೂರ್ಣಾಙ್ಕ slot = (inq_head + inq_ಎಣಿಕೆ) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_ಎಣಿಕೆ++;
        } ಅನ್ಯಥಾ {
            send_busy[(ಪೂರ್ಣಾಙ್ಕ)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    ಪ್ರತ್ಯಾಗಮ n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
ತಾರ್ಕಿಕ apple_rdma::impl::acquire_pending() {
    ಯದಿ (pend_buf >= 0) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    ಪ್ರತಿ (;;) {
        ಯದಿ (broken) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < RDMA_NBUF; k++) ಯದಿ (!send_busy[k]) { pend_buf = k; pend_len = 0; ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ; }
        ಯದಿ (progress() < 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
ತಾರ್ಕಿಕ apple_rdma::impl::post_pending() {
    ಯದಿ (pend_buf < 0) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    ಪೂರ್ಣಾಙ್ಕ i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    ಯದಿ (!post_send(i, RDMA_STRIDE)) { broken = ಸತ್ಯ; ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
ತಾರ್ಕಿಕ apple_rdma::send(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size) {
    impl * c = pimpl.get();
    ಧ್ರುವ uint8_t * p = (ಧ್ರುವ uint8_t *)data;
    ಯಾವತ್ (size > 0) {
        ಯದಿ (c->broken) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (!c->acquire_pending()) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + ಪರಿಮಾಣ(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        ಯದಿ (c->pend_len == RDMA_PAYLOAD) { ಯದಿ (!c->post_pending()) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ apple_rdma::recv(ಶೂನ್ಯ * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    ಯದಿ (!c->post_pending()) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    ಯಾವತ್ (size > 0) {
        ಯದಿ (c->inq_ಎಣಿಕೆ == 0) {
            ಯದಿ (c->broken) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            ಪೂರ್ಣಾಙ್ಕ n = c->progress();
            ಯದಿ (n < 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            ಯದಿ (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                ಯದಿ ((++idle & 0xFFFFF) == 0) {
                    ಸಂರಚನಾ pollfd pfd = { c->fd, POLLIN, 0 };
                    ಯದಿ (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                    }
                }
            } ಅನ್ಯಥಾ {
                idle = 0;
            }
            ಅನುವರ್ತ;
        }
        idle = 0;
        ಪೂರ್ಣಾಙ್ಕ slot = c->inq_head;
        ಪೂರ್ಣಾಙ್ಕ b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + ಪರಿಮಾಣ(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        ಯದಿ (c->inq[slot].off == c->inq[slot].len) {
            ಯದಿ (!c->post_recv(b)) { c->broken = ಸತ್ಯ; ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_ಎಣಿಕೆ--;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ apple_rdma::flush() {
    ಪ್ರತ್ಯಾಗಮ pimpl->post_pending();
}

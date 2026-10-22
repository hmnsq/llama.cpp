#ضمن "transport-apple.h"
#ضمن "transport.h"
#ضمن "log.h"

#ضمن <infiniband/verbs.h>

#ضمن <cerrno>
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <string>
#ضمن <dlfcn.h>
#ضمن <poll.h>
#ضمن <sys/socket.h>
#ضمن <unistd.h>

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
ساكن constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
ساكن constexpr صحيح      RDMA_NBUF        = 16;          // ring depth (frames per direction)
ساكن constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
ساكن constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
ساكن constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
ساكن constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
ساكن constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
ساكن constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
ساكن constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
ساكن constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

بنية rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
ساكن constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - حجم(rdma_seg_hdr);

بنية apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(حجم(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

بنية apple_rdma::impl {
    صحيح fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    بنية ibv_context * ctx = nullptr;
    بنية ibv_pd * pd = nullptr;
    بنية ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    بنية ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    بنية ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    بنية ibv_mr * recv_mr  = nullptr;

    صحيح      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    بنية { صحيح buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    صحيح      inq_head = 0;
    صحيح      inq_count = 0;
    صحيح      pend_buf = -1;
    uint32_t pend_len = 0;
    منطقي     broken = ܕܓܠܐ;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    صحيح          gid_idx = 0;
    تعداد ibv_mtu path_mtu = IBV_MTU_1024;

    صحيح  progress();
    منطقي acquire_pending();
    منطقي post_pending();

    منطقي post_recv(صحيح i) {
        بنية ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        بنية ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        ܦܢܐ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    منطقي post_send(صحيح i, size_t len) {
        بنية ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        بنية ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        ܦܢܐ ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = ܫܪܪܐ;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        ܐܢ (qp) ibv_destroy_qp(qp);
        ܐܢ (send_mr) ibv_dereg_mr(send_mr);
        ܐܢ (recv_mr) ibv_dereg_mr(recv_mr);
        حرر(send_mem);
        حرر(recv_mem);
        ܐܢ (cq)  ibv_destroy_cq(cq);
        ܐܢ (pd)  ibv_dealloc_pd(pd);
        ܐܢ (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = افتراضي;

منطقي apple_rdma::broken() ثابت {
    ܦܢܐ pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
ساكن منطقي tcp_send_byte(صحيح fd, uint8_t b) {
    ssize_t n;
    ܥܒܕ { n = ::send(fd, &b, حجم(b), 0); } ܟܡܐ (n < 0 && errno == EINTR);
    ܦܢܐ n == حجم(b);
}

ساكن منطقي tcp_recv_byte(صحيح fd, uint8_t * b) {
    ssize_t n;
    ܥܒܕ { n = ::recv(fd, b, حجم(*b), 0); } ܟܡܐ (n < 0 && errno == EINTR);
    ܦܢܐ n == (ssize_t)حجم(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
ساكن صحيح rdma_match_gid(بنية ibv_context * ctx, uint8_t port, صحيح gid_tbl_len,
                          ثابت uint8_t * target, اتحاد ibv_gid * out) {
    لكل (صحيح i = 0; i < gid_tbl_len; i++) {
        اتحاد ibv_gid g;
        ܐܢ (ibv_query_gid(ctx, port, i, &g) != 0) استمر;
        ܐܢ (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) استمر;
        ܐܢ (out) *out = g;
        ܦܢܐ i;
    }
    ܦܢܐ -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
ساكن uint8_t rdma_first_active_port(بنية ibv_context * ctx, بنية ibv_port_attr * out) {
    بنية ibv_device_attr da;
    ܐܢ (ibv_query_device(ctx, &da) != 0) ܦܢܐ 0;
    لكل (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        بنية ibv_port_attr pa;
        ܐܢ (ibv_query_port(ctx, p, &pa) != 0) استمر;
        ܐܢ (pa.state == IBV_PORT_ACTIVE) { ܐܢ (out) *out = pa; ܦܢܐ p; }
    }
    ܦܢܐ 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
ساكن منطقي rdma_library_present() {
    ساكن ثابت منطقي present = [] {
        فراغ * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        ܐܢ (handle == nullptr) {
            ܦܢܐ ܕܓܠܐ;
        }
        dlclose(handle);
        ܦܢܐ ܫܪܪܐ;
    }();
    ܦܢܐ present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(صحيح fd, ثابت uint8_t * target_gid, uint8_t * caps) {
    ܐܢ (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        ܦܢܐ nullptr;
    }
    صحيح ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    ܐܢ (!devs) ܦܢܐ nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    بنية ibv_port_attr pa = {};
    اتحاد ibv_gid gid = {};
    صحيح gid_idx = -1;
    std::string matched;
    لكل (صحيح d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        ܐܢ (!c) استمر;
        بنية ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        صحيح gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        ܐܢ (gi < 0) { ibv_close_device(c); استمر; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        ثابت حرف * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        اكسر;
    }
    ibv_free_device_list(devs);
    ܐܢ (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        ܦܢܐ nullptr;
    }

    std::unique_ptr<impl> c(جديد impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    ܐܢ (!c->pd) ܦܢܐ nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    ܐܢ (!c->cq) ܦܢܐ nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    ܐܢ (!c->qp) ܦܢܐ nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        ܐܢ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            ܦܢܐ nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    ܐܢ (page <= 0) page = 4096;
    ثابت size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    ܐܢ (posix_memalign((فراغ **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    ܐܢ (posix_memalign((فراغ **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    ܐܢ (!c->send_mem || !c->recv_mem) ܦܢܐ nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    ثابت صحيح mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    ܐܢ (!c->send_mr || !c->recv_mr) ܦܢܐ nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_num;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, حجم(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    ܦܢܐ std::unique_ptr<apple_rdma>(جديد apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
منطقي apple_rdma::activate(ثابت uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, حجم(rc));

    منطقي ok = ܫܪܪܐ;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_num = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_num      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        ܐܢ (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = ܕܓܠܐ;
        }
    }
    ܐܢ (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        ܐܢ (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = ܕܓܠܐ;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    لكل (صحيح i = 0; ok && i < RDMA_NBUF; i++) {
        ܐܢ (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = ܕܓܠܐ;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    ܐܢ (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        ܦܢܐ ܕܓܠܐ;
    }
    ܐܢ (!ok || peer_ready != RDMA_SYNC_READY) {
        ܦܢܐ ܕܓܠܐ;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    ܦܢܐ ܫܪܪܐ;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
صحيح apple_rdma::impl::progress() {
    بنية ibv_wc wc[RDMA_NBUF * 2];
    صحيح n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    ܐܢ (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = ܫܪܪܐ; ܦܢܐ -1; }
    لكل (صحيح j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        منطقي is_recv = (id & RDMA_RECV_WR) != 0;
        ܐܢ (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = ܫܪܪܐ;
            ܦܢܐ -1;
        }
        ܐܢ (is_recv) {
            صحيح b = (صحيح)(id & RDMA_WR_IDX_MASK);
            ثابت rdma_seg_hdr * h = (ثابت rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            ܐܢ (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = ܫܪܪܐ; ܦܢܐ -1; }
            ܐܢ (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = ܫܪܪܐ; ܦܢܐ -1; }
            صحيح slot = (inq_head + inq_count) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_count++;
        } ܐܚܪܢܐ {
            send_busy[(صحيح)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    ܦܢܐ n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
منطقي apple_rdma::impl::acquire_pending() {
    ܐܢ (pend_buf >= 0) ܦܢܐ ܫܪܪܐ;
    لكل (;;) {
        ܐܢ (broken) ܦܢܐ ܕܓܠܐ;
        لكل (صحيح k = 0; k < RDMA_NBUF; k++) ܐܢ (!send_busy[k]) { pend_buf = k; pend_len = 0; ܦܢܐ ܫܪܪܐ; }
        ܐܢ (progress() < 0) ܦܢܐ ܕܓܠܐ;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
منطقي apple_rdma::impl::post_pending() {
    ܐܢ (pend_buf < 0) ܦܢܐ ܫܪܪܐ;
    صحيح i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    ܐܢ (!post_send(i, RDMA_STRIDE)) { broken = ܫܪܪܐ; ܦܢܐ ܕܓܠܐ; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    ܦܢܐ ܫܪܪܐ;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
منطقي apple_rdma::send(ثابت فراغ * data, size_t size) {
    impl * c = pimpl.get();
    ثابت uint8_t * p = (ثابت uint8_t *)data;
    ܟܡܐ (size > 0) {
        ܐܢ (c->broken) ܦܢܐ ܕܓܠܐ;
        ܐܢ (!c->acquire_pending()) ܦܢܐ ܕܓܠܐ;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + حجم(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        ܐܢ (c->pend_len == RDMA_PAYLOAD) { ܐܢ (!c->post_pending()) ܦܢܐ ܕܓܠܐ; }
    }
    ܦܢܐ ܫܪܪܐ;
}

منطقي apple_rdma::recv(فراغ * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    ܐܢ (!c->post_pending()) ܦܢܐ ܕܓܠܐ;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    ܟܡܐ (size > 0) {
        ܐܢ (c->inq_count == 0) {
            ܐܢ (c->broken) ܦܢܐ ܕܓܠܐ;
            صحيح n = c->progress();
            ܐܢ (n < 0) ܦܢܐ ܕܓܠܐ;
            ܐܢ (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                ܐܢ ((++idle & 0xFFFFF) == 0) {
                    بنية pollfd pfd = { c->fd, POLLIN, 0 };
                    ܐܢ (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        ܦܢܐ ܕܓܠܐ;
                    }
                }
            } ܐܚܪܢܐ {
                idle = 0;
            }
            استمر;
        }
        idle = 0;
        صحيح slot = c->inq_head;
        صحيح b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + حجم(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        ܐܢ (c->inq[slot].off == c->inq[slot].len) {
            ܐܢ (!c->post_recv(b)) { c->broken = ܫܪܪܐ; ܦܢܐ ܕܓܠܐ; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_count--;
        }
    }
    ܦܢܐ ܫܪܪܐ;
}

منطقي apple_rdma::flush() {
    ܦܢܐ pimpl->post_pending();
}

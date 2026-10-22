#অন্তর্ভাব "transport-apple.h"
#অন্তর্ভাব "transport.h"
#অন্তর্ভাব "log.h"

#অন্তর্ভাব <infiniband/verbs.h>

#অন্তর্ভাব <cerrno>
#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <string>
#অন্তর্ভাব <dlfcn.h>
#অন্তর্ভাব <poll.h>
#অন্তর্ভাব <sys/socket.h>
#অন্তর্ভাব <unistd.h>

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
স্থির constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
স্থির constexpr পূর্ণাঙ্ক      RDMA_NBUF        = 16;          // ring depth (frames per direction)
স্থির constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
স্থির constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
স্থির constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
স্থির constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
স্থির constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
স্থির constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
স্থির constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
স্থির constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

সংরচনা rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
স্থির constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - পরিমাণ(rdma_seg_hdr);

সংরচনা apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(পরিমাণ(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

সংরচনা apple_rdma::impl {
    পূর্ণাঙ্ক fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    সংরচনা ibv_context * ctx = nullptr;
    সংরচনা ibv_pd * pd = nullptr;
    সংরচনা ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    সংরচনা ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    সংরচনা ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    সংরচনা ibv_mr * recv_mr  = nullptr;

    পূর্ণাঙ্ক      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    সংরচনা { পূর্ণাঙ্ক buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    পূর্ণাঙ্ক      inq_head = 0;
    পূর্ণাঙ্ক      inq_গুনতি = 0;
    পূর্ণাঙ্ক      pend_buf = -1;
    uint32_t pend_len = 0;
    তার্কিক     broken = অসত্য;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    পূর্ণাঙ্ক          gid_idx = 0;
    গণনা ibv_mtu path_mtu = IBV_MTU_1024;

    পূর্ণাঙ্ক  progress();
    তার্কিক acquire_pending();
    তার্কিক post_pending();

    তার্কিক post_recv(পূর্ণাঙ্ক i) {
        সংরচনা ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        সংরচনা ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.অঙ্ক_sge = 1;
        প্রত্যাগম ibv_post_recv(qp, &wr, &bad) == 0;
    }

    তার্কিক post_send(পূর্ণাঙ্ক i, size_t len) {
        সংরচনা ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        সংরচনা ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.অঙ্ক_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        প্রত্যাগম ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = সত্য;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        যদি (qp) ibv_destroy_qp(qp);
        যদি (send_mr) ibv_dereg_mr(send_mr);
        যদি (recv_mr) ibv_dereg_mr(recv_mr);
        মুক্ত(send_mem);
        মুক্ত(recv_mem);
        যদি (cq)  ibv_destroy_cq(cq);
        যদি (pd)  ibv_dealloc_pd(pd);
        যদি (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = স্বভাব;

তার্কিক apple_rdma::broken() ধ্রুব {
    প্রত্যাগম pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
স্থির তার্কিক tcp_send_byte(পূর্ণাঙ্ক fd, uint8_t b) {
    ssize_t n;
    কুরু { n = ::send(fd, &b, পরিমাণ(b), 0); } যাবত্ (n < 0 && errno == EINTR);
    প্রত্যাগম n == পরিমাণ(b);
}

স্থির তার্কিক tcp_recv_byte(পূর্ণাঙ্ক fd, uint8_t * b) {
    ssize_t n;
    কুরু { n = ::recv(fd, b, পরিমাণ(*b), 0); } যাবত্ (n < 0 && errno == EINTR);
    প্রত্যাগম n == (ssize_t)পরিমাণ(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
স্থির পূর্ণাঙ্ক rdma_match_gid(সংরচনা ibv_context * ctx, uint8_t port, পূর্ণাঙ্ক gid_tbl_len,
                          ধ্রুব uint8_t * target, সঙ্ঘ ibv_gid * out) {
    প্রতি (পূর্ণাঙ্ক i = 0; i < gid_tbl_len; i++) {
        সঙ্ঘ ibv_gid g;
        যদি (ibv_query_gid(ctx, port, i, &g) != 0) অনুবর্ত;
        যদি (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) অনুবর্ত;
        যদি (out) *out = g;
        প্রত্যাগম i;
    }
    প্রত্যাগম -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
স্থির uint8_t rdma_first_active_port(সংরচনা ibv_context * ctx, সংরচনা ibv_port_attr * out) {
    সংরচনা ibv_device_attr da;
    যদি (ibv_query_device(ctx, &da) != 0) প্রত্যাগম 0;
    প্রতি (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        সংরচনা ibv_port_attr pa;
        যদি (ibv_query_port(ctx, p, &pa) != 0) অনুবর্ত;
        যদি (pa.state == IBV_PORT_ACTIVE) { যদি (out) *out = pa; প্রত্যাগম p; }
    }
    প্রত্যাগম 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
স্থির তার্কিক rdma_library_present() {
    স্থির ধ্রুব তার্কিক present = [] {
        শূন্য * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        যদি (handle == nullptr) {
            প্রত্যাগম অসত্য;
        }
        dlclose(handle);
        প্রত্যাগম সত্য;
    }();
    প্রত্যাগম present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(পূর্ণাঙ্ক fd, ধ্রুব uint8_t * target_gid, uint8_t * caps) {
    যদি (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        প্রত্যাগম nullptr;
    }
    পূর্ণাঙ্ক ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    যদি (!devs) প্রত্যাগম nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    সংরচনা ibv_port_attr pa = {};
    সঙ্ঘ ibv_gid gid = {};
    পূর্ণাঙ্ক gid_idx = -1;
    std::string matched;
    প্রতি (পূর্ণাঙ্ক d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        যদি (!c) অনুবর্ত;
        সংরচনা ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        পূর্ণাঙ্ক gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        যদি (gi < 0) { ibv_close_device(c); অনুবর্ত; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        ধ্রুব বর্ণ * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        ভঙ্গ;
    }
    ibv_free_device_list(devs);
    যদি (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        প্রত্যাগম nullptr;
    }

    std::unique_ptr<impl> c(নূতন impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    যদি (!c->pd) প্রত্যাগম nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    যদি (!c->cq) প্রত্যাগম nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    যদি (!c->qp) প্রত্যাগম nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_অঙ্ক = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        যদি (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            প্রত্যাগম nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    যদি (page <= 0) page = 4096;
    ধ্রুব size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    যদি (posix_memalign((শূন্য **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    যদি (posix_memalign((শূন্য **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    যদি (!c->send_mem || !c->recv_mem) প্রত্যাগম nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    ধ্রুব পূর্ণাঙ্ক mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    যদি (!c->send_mr || !c->recv_mr) প্রত্যাগম nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_অঙ্ক;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, পরিমাণ(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    প্রত্যাগম std::unique_ptr<apple_rdma>(নূতন apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
তার্কিক apple_rdma::activate(ধ্রুব uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, পরিমাণ(rc));

    তার্কিক ok = সত্য;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_অঙ্ক = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_অঙ্ক      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        যদি (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = অসত্য;
        }
    }
    যদি (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        যদি (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = অসত্য;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    প্রতি (পূর্ণাঙ্ক i = 0; ok && i < RDMA_NBUF; i++) {
        যদি (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = অসত্য;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    যদি (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        প্রত্যাগম অসত্য;
    }
    যদি (!ok || peer_ready != RDMA_SYNC_READY) {
        প্রত্যাগম অসত্য;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    প্রত্যাগম সত্য;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
পূর্ণাঙ্ক apple_rdma::impl::progress() {
    সংরচনা ibv_wc wc[RDMA_NBUF * 2];
    পূর্ণাঙ্ক n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    যদি (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = সত্য; প্রত্যাগম -1; }
    প্রতি (পূর্ণাঙ্ক j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        তার্কিক is_recv = (id & RDMA_RECV_WR) != 0;
        যদি (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = সত্য;
            প্রত্যাগম -1;
        }
        যদি (is_recv) {
            পূর্ণাঙ্ক b = (পূর্ণাঙ্ক)(id & RDMA_WR_IDX_MASK);
            ধ্রুব rdma_seg_hdr * h = (ধ্রুব rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            যদি (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = সত্য; প্রত্যাগম -1; }
            যদি (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = সত্য; প্রত্যাগম -1; }
            পূর্ণাঙ্ক slot = (inq_head + inq_গুনতি) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_গুনতি++;
        } অন্যথা {
            send_busy[(পূর্ণাঙ্ক)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    প্রত্যাগম n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
তার্কিক apple_rdma::impl::acquire_pending() {
    যদি (pend_buf >= 0) প্রত্যাগম সত্য;
    প্রতি (;;) {
        যদি (broken) প্রত্যাগম অসত্য;
        প্রতি (পূর্ণাঙ্ক k = 0; k < RDMA_NBUF; k++) যদি (!send_busy[k]) { pend_buf = k; pend_len = 0; প্রত্যাগম সত্য; }
        যদি (progress() < 0) প্রত্যাগম অসত্য;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
তার্কিক apple_rdma::impl::post_pending() {
    যদি (pend_buf < 0) প্রত্যাগম সত্য;
    পূর্ণাঙ্ক i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    যদি (!post_send(i, RDMA_STRIDE)) { broken = সত্য; প্রত্যাগম অসত্য; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    প্রত্যাগম সত্য;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
তার্কিক apple_rdma::send(ধ্রুব শূন্য * data, size_t size) {
    impl * c = pimpl.get();
    ধ্রুব uint8_t * p = (ধ্রুব uint8_t *)data;
    যাবত্ (size > 0) {
        যদি (c->broken) প্রত্যাগম অসত্য;
        যদি (!c->acquire_pending()) প্রত্যাগম অসত্য;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + পরিমাণ(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        যদি (c->pend_len == RDMA_PAYLOAD) { যদি (!c->post_pending()) প্রত্যাগম অসত্য; }
    }
    প্রত্যাগম সত্য;
}

তার্কিক apple_rdma::recv(শূন্য * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    যদি (!c->post_pending()) প্রত্যাগম অসত্য;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    যাবত্ (size > 0) {
        যদি (c->inq_গুনতি == 0) {
            যদি (c->broken) প্রত্যাগম অসত্য;
            পূর্ণাঙ্ক n = c->progress();
            যদি (n < 0) প্রত্যাগম অসত্য;
            যদি (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                যদি ((++idle & 0xFFFFF) == 0) {
                    সংরচনা pollfd pfd = { c->fd, POLLIN, 0 };
                    যদি (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        প্রত্যাগম অসত্য;
                    }
                }
            } অন্যথা {
                idle = 0;
            }
            অনুবর্ত;
        }
        idle = 0;
        পূর্ণাঙ্ক slot = c->inq_head;
        পূর্ণাঙ্ক b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + পরিমাণ(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        যদি (c->inq[slot].off == c->inq[slot].len) {
            যদি (!c->post_recv(b)) { c->broken = সত্য; প্রত্যাগম অসত্য; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_গুনতি--;
        }
    }
    প্রত্যাগম সত্য;
}

তার্কিক apple_rdma::flush() {
    প্রত্যাগম pimpl->post_pending();
}

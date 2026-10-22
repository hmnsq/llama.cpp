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
    صحيح      inq_ژماردن = 0;
    صحيح      pend_buf = -1;
    uint32_t pend_len = 0;
    منطقي     broken = خطأ;

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
        wr.ژمارە_sge = 1;
        أرجع ibv_post_recv(qp, &wr, &bad) == 0;
    }

    منطقي post_send(صحيح i, size_t len) {
        بنية ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        بنية ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ژمارە_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        أرجع ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = true;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        إذا (qp) ibv_destroy_qp(qp);
        إذا (send_mr) ibv_dereg_mr(send_mr);
        إذا (recv_mr) ibv_dereg_mr(recv_mr);
        حرر(send_mem);
        حرر(recv_mem);
        إذا (cq)  ibv_destroy_cq(cq);
        إذا (pd)  ibv_dealloc_pd(pd);
        إذا (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = افتراضي;

منطقي apple_rdma::broken() ثابت {
    أرجع pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
ساكن منطقي tcp_send_byte(صحيح fd, uint8_t b) {
    ssize_t n;
    افعل { n = ::send(fd, &b, حجم(b), 0); } طالما (n < 0 && errno == EINTR);
    أرجع n == حجم(b);
}

ساكن منطقي tcp_recv_byte(صحيح fd, uint8_t * b) {
    ssize_t n;
    افعل { n = ::recv(fd, b, حجم(*b), 0); } طالما (n < 0 && errno == EINTR);
    أرجع n == (ssize_t)حجم(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
ساكن صحيح rdma_match_gid(بنية ibv_context * ctx, uint8_t port, صحيح gid_tbl_len,
                          ثابت uint8_t * target, اتحاد ibv_gid * out) {
    لكل (صحيح i = 0; i < gid_tbl_len; i++) {
        اتحاد ibv_gid g;
        إذا (ibv_query_gid(ctx, port, i, &g) != 0) استمر;
        إذا (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) استمر;
        إذا (out) *out = g;
        أرجع i;
    }
    أرجع -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
ساكن uint8_t rdma_first_active_port(بنية ibv_context * ctx, بنية ibv_port_attr * out) {
    بنية ibv_device_attr da;
    إذا (ibv_query_device(ctx, &da) != 0) أرجع 0;
    لكل (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        بنية ibv_port_attr pa;
        إذا (ibv_query_port(ctx, p, &pa) != 0) استمر;
        إذا (pa.state == IBV_PORT_ACTIVE) { إذا (out) *out = pa; أرجع p; }
    }
    أرجع 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
ساكن منطقي rdma_library_present() {
    ساكن ثابت منطقي present = [] {
        فراغ * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        إذا (handle == nullptr) {
            أرجع خطأ;
        }
        dlclose(handle);
        أرجع true;
    }();
    أرجع present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(صحيح fd, ثابت uint8_t * target_gid, uint8_t * caps) {
    إذا (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        أرجع nullptr;
    }
    صحيح ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    إذا (!devs) أرجع nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    بنية ibv_port_attr pa = {};
    اتحاد ibv_gid gid = {};
    صحيح gid_idx = -1;
    std::string matched;
    لكل (صحيح d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        إذا (!c) استمر;
        بنية ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        صحيح gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        إذا (gi < 0) { ibv_close_device(c); استمر; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        ثابت حرف * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        اكسر;
    }
    ibv_free_device_list(devs);
    إذا (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        أرجع nullptr;
    }

    std::unique_ptr<impl> c(جديد impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    إذا (!c->pd) أرجع nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    إذا (!c->cq) أرجع nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    إذا (!c->qp) أرجع nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_ژمارە = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        إذا (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            أرجع nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    إذا (page <= 0) page = 4096;
    ثابت size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    إذا (posix_memalign((فراغ **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    إذا (posix_memalign((فراغ **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    إذا (!c->send_mem || !c->recv_mem) أرجع nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    ثابت صحيح mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    إذا (!c->send_mr || !c->recv_mr) أرجع nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_ژمارە;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, حجم(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    أرجع std::unique_ptr<apple_rdma>(جديد apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
منطقي apple_rdma::activate(ثابت uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, حجم(rc));

    منطقي ok = true;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.path_mtu   = c->path_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_ژمارە = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.port_ژمارە      = c->port;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = (uint8_t)c->gid_idx;
        memcpy(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_SIZE);
        إذا (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = خطأ;
        }
    }
    إذا (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        إذا (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = خطأ;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    لكل (صحيح i = 0; ok && i < RDMA_NBUF; i++) {
        إذا (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = خطأ;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    إذا (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        أرجع خطأ;
    }
    إذا (!ok || peer_ready != RDMA_SYNC_READY) {
        أرجع خطأ;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    أرجع true;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
صحيح apple_rdma::impl::progress() {
    بنية ibv_wc wc[RDMA_NBUF * 2];
    صحيح n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    إذا (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = true; أرجع -1; }
    لكل (صحيح j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        منطقي is_recv = (id & RDMA_RECV_WR) != 0;
        إذا (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = true;
            أرجع -1;
        }
        إذا (is_recv) {
            صحيح b = (صحيح)(id & RDMA_WR_IDX_MASK);
            ثابت rdma_seg_hdr * h = (ثابت rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            إذا (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = true; أرجع -1; }
            إذا (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = true; أرجع -1; }
            صحيح slot = (inq_head + inq_ژماردن) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_ژماردن++;
        } وإلا {
            send_busy[(صحيح)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    أرجع n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
منطقي apple_rdma::impl::acquire_pending() {
    إذا (pend_buf >= 0) أرجع true;
    لكل (;;) {
        إذا (broken) أرجع خطأ;
        لكل (صحيح k = 0; k < RDMA_NBUF; k++) إذا (!send_busy[k]) { pend_buf = k; pend_len = 0; أرجع true; }
        إذا (progress() < 0) أرجع خطأ;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
منطقي apple_rdma::impl::post_pending() {
    إذا (pend_buf < 0) أرجع true;
    صحيح i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    إذا (!post_send(i, RDMA_STRIDE)) { broken = true; أرجع خطأ; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    أرجع true;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
منطقي apple_rdma::send(ثابت فراغ * data, size_t size) {
    impl * c = pimpl.get();
    ثابت uint8_t * p = (ثابت uint8_t *)data;
    طالما (size > 0) {
        إذا (c->broken) أرجع خطأ;
        إذا (!c->acquire_pending()) أرجع خطأ;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + حجم(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        إذا (c->pend_len == RDMA_PAYLOAD) { إذا (!c->post_pending()) أرجع خطأ; }
    }
    أرجع true;
}

منطقي apple_rdma::recv(فراغ * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    إذا (!c->post_pending()) أرجع خطأ;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    طالما (size > 0) {
        إذا (c->inq_ژماردن == 0) {
            إذا (c->broken) أرجع خطأ;
            صحيح n = c->progress();
            إذا (n < 0) أرجع خطأ;
            إذا (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                إذا ((++idle & 0xFFFFF) == 0) {
                    بنية pollfd pfd = { c->fd, POLLIN, 0 };
                    إذا (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        أرجع خطأ;
                    }
                }
            } وإلا {
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
        إذا (c->inq[slot].off == c->inq[slot].len) {
            إذا (!c->post_recv(b)) { c->broken = true; أرجع خطأ; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_ژماردن--;
        }
    }
    أرجع true;
}

منطقي apple_rdma::flush() {
    أرجع pimpl->post_pending();
}

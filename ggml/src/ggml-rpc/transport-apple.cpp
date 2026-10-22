#شامل "transport-apple.h"
#شامل "transport.h"
#شامل "log.h"

#شامل <infiniband/verbs.h>

#شامل <cerrno>
#شامل <cstdlib>
#شامل <cstring>
#شامل <string>
#شامل <dlfcn.h>
#شامل <poll.h>
#شامل <sys/socket.h>
#شامل <unistd.h>

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
ساکن constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
ساکن constexpr صحیح      RDMA_NBUF        = 16;          // ring depth (frames per direction)
ساکن constexpr size_t   RDMA_چوکھٹا       = 4096;        // Thunderbolt frame (fixed on Apple)
ساکن constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
ساکن constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
ساکن constexpr size_t   RDMA_GID_سائز    = 16;

static_assert(RDMA_STRIDE % RDMA_چوکھٹا == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
ساکن constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_چوکھٹا);
ساکن constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
ساکن constexpr uint64_t RDMA_WR_IDX_نقاب = 0xffff;      // buffer index in the low bits of wr_id
ساکن constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

ساخت rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
ساکن constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - حجم(rdma_seg_hdr);

ساخت apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_سائز];
};

static_assert(حجم(apple_rdma_caps) == RPC_CONN_CAPS_سائز, "apple_rdma_caps must match conn_caps size");

ساخت apple_rdma::impl {
    صحیح fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    ساخت ibv_context * ctx = خالی_اشاریہ;
    ساخت ibv_pd * pd = خالی_اشاریہ;
    ساخت ibv_cq * cq = خالی_اشاریہ;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    ساخت ibv_qp * qp = خالی_اشاریہ;

    uint8_t       * send_mem = خالی_اشاریہ;
    ساخت ibv_mr * send_mr  = خالی_اشاریہ;
    uint8_t       * recv_mem = خالی_اشاریہ;
    ساخت ibv_mr * recv_mr  = خالی_اشاریہ;

    صحیح      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    ساخت { صحیح buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    صحیح      inq_سر = 0;
    صحیح      inq_گنتی = 0;
    صحیح      pend_buf = -1;
    uint32_t pend_len = 0;
    منطقی     broken = جھوٹ;

    uint32_t     qpn = 0;
    uint8_t      پورٹ = 0;
    صحیح          gid_idx = 0;
    شمار ibv_mtu راستہ_mtu = IBV_MTU_1024;

    صحیح  پیش_رفت();
    منطقی acquire_pending();
    منطقی post_pending();

    منطقی post_recv(صحیح i) {
        ساخت ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.لمبائی = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        ساخت ibv_recv_wr wr = {}, * bad = خالی_اشاریہ;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.عدد_sge = 1;
        واپس ibv_post_recv(qp, &wr, &bad) == 0;
    }

    منطقی post_send(صحیح i, size_t len) {
        ساخت ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.لمبائی = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        ساخت ibv_send_wr wr = {}, * bad = خالی_اشاریہ;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.عدد_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        واپس ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = سچ;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        اگر (qp) ibv_destroy_qp(qp);
        اگر (send_mr) ibv_dereg_mr(send_mr);
        اگر (recv_mr) ibv_dereg_mr(recv_mr);
        آزاد(send_mem);
        آزاد(recv_mem);
        اگر (cq)  ibv_destroy_cq(cq);
        اگر (pd)  ibv_dealloc_pd(pd);
        اگر (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = طےشدہ;

منطقی apple_rdma::broken() ثابت {
    واپس pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
ساکن منطقی tcp_send_byte(صحیح fd, uint8_t b) {
    ssize_t n;
    کرو { n = ::send(fd, &b, حجم(b), 0); } جب (n < 0 && errno == EINTR);
    واپس n == حجم(b);
}

ساکن منطقی tcp_recv_byte(صحیح fd, uint8_t * b) {
    ssize_t n;
    کرو { n = ::recv(fd, b, حجم(*b), 0); } جب (n < 0 && errno == EINTR);
    واپس n == (ssize_t)حجم(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
ساکن صحیح rdma_مطابقت_gid(ساخت ibv_context * ctx, uint8_t پورٹ, صحیح gid_tbl_len,
                          ثابت uint8_t * ہدف, اتحاد ibv_gid * برآمد) {
    برائے (صحیح i = 0; i < gid_tbl_len; i++) {
        اتحاد ibv_gid g;
        اگر (ibv_query_gid(ctx, پورٹ, i, &g) != 0) جاری;
        اگر (memcmp(g.raw, ہدف, RDMA_GID_سائز) != 0) جاری;
        اگر (برآمد) *برآمد = g;
        واپس i;
    }
    واپس -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
ساکن uint8_t rdma_پہلا_فعال_پورٹ(ساخت ibv_context * ctx, ساخت ibv_پورٹ_attr * برآمد) {
    ساخت ibv_device_attr da;
    اگر (ibv_query_device(ctx, &da) != 0) واپس 0;
    برائے (uint8_t p = 1; p <= da.phys_پورٹ_cnt; p++) {
        ساخت ibv_پورٹ_attr pa;
        اگر (ibv_query_پورٹ(ctx, p, &pa) != 0) جاری;
        اگر (pa.state == IBV_پورٹ_فعال) { اگر (برآمد) *برآمد = pa; واپس p; }
    }
    واپس 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
ساکن منطقی rdma_library_present() {
    ساکن ثابت منطقی present = [] {
        خالی * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        اگر (handle == خالی_اشاریہ) {
            واپس جھوٹ;
        }
        dlclose(handle);
        واپس سچ;
    }();
    واپس present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(صحیح fd, ثابت uint8_t * ہدف_gid, uint8_t * caps) {
    اگر (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        واپس خالی_اشاریہ;
    }
    صحیح ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    اگر (!devs) واپس خالی_اشاریہ;

    ibv_context * ctx = خالی_اشاریہ;
    uint8_t پورٹ = 0;
    ساخت ibv_پورٹ_attr pa = {};
    اتحاد ibv_gid gid = {};
    صحیح gid_idx = -1;
    std::string matched;
    برائے (صحیح d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        اگر (!c) جاری;
        ساخت ibv_پورٹ_attr p = {};
        uint8_t pt = rdma_پہلا_فعال_پورٹ(c, &p);
        صحیح gi = pt ? rdma_مطابقت_gid(c, pt, p.gid_tbl_len, ہدف_gid, &gid) : -1;
        اگر (gi < 0) { ibv_close_device(c); جاری; }
        ctx = c; پورٹ = pt; pa = p; gid_idx = gi;
        ثابت حرف * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        توڑو;
    }
    ibv_free_device_list(devs);
    اگر (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        واپس خالی_اشاریہ;
    }

    std::unique_ptr<impl> c(نیا impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->پورٹ = پورٹ;
    c->gid_idx = gid_idx;
    c->راستہ_mtu = pa.فعال_mtu;

    c->pd = ibv_alloc_pd(ctx);
    اگر (!c->pd) واپس خالی_اشاریہ;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, خالی_اشاریہ, خالی_اشاریہ, 0);
    اگر (!c->cq) واپس خالی_اشاریہ;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    اگر (!c->qp) واپس خالی_اشاریہ;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_اشاریہ = 0;
        a.پورٹ_عدد = پورٹ;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        اگر (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_اشاریہ | IBV_QP_پورٹ | IBV_QP_ACCESS_FLAGS) != 0) {
            واپس خالی_اشاریہ;
        }
    }

    طویل صفحہ = sysconf(_SC_PAGESIZE);
    اگر (صفحہ <= 0) صفحہ = 4096;
    ثابت size_t ring_بائٹیں = (size_t)RDMA_NBUF * RDMA_STRIDE;
    اگر (posix_memalign((خالی **)&c->send_mem, (size_t)صفحہ, ring_بائٹیں) != 0) c->send_mem = خالی_اشاریہ;
    اگر (posix_memalign((خالی **)&c->recv_mem, (size_t)صفحہ, ring_بائٹیں) != 0) c->recv_mem = خالی_اشاریہ;
    اگر (!c->send_mem || !c->recv_mem) واپس خالی_اشاریہ;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    ثابت صحیح mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_بائٹیں, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_بائٹیں, mr_flags);
    اگر (!c->send_mr || !c->recv_mr) واپس خالی_اشاریہ;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_عدد;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    حافظہ_نقل(rc.gid, gid.raw, RDMA_GID_سائز);
    حافظہ_نقل(caps, &rc, حجم(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), پورٹ, gid_idx, c->qpn, (غیرعلامتی)pa.lid, 128 << c->راستہ_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    واپس std::unique_ptr<apple_rdma>(نیا apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
منطقی apple_rdma::activate(ثابت uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    حافظہ_نقل(&rc, caps, حجم(rc));

    منطقی ok = سچ;
    {
        ibv_qp_attr a = {};
        a.qp_state   = IBV_QPS_RTR;
        a.راستہ_mtu   = c->راستہ_mtu;
        a.rq_psn     = RDMA_PSN;
        a.dest_qp_عدد = rc.qpn;
        a.ah_attr.is_global     = 1;
        a.ah_attr.پورٹ_عدد      = c->پورٹ;
        a.ah_attr.sl            = 0;
        a.ah_attr.src_راستہ_بٹ = 0;
        a.ah_attr.dlid          = rc.lid;
        a.ah_attr.grh.hop_حد_بندی  = 1;
        a.ah_attr.grh.sgid_اشاریہ = (uint8_t)c->gid_idx;
        حافظہ_نقل(&a.ah_attr.grh.dgid, rc.gid, RDMA_GID_سائز);
        اگر (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_راستہ_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = جھوٹ;
        }
    }
    اگر (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        اگر (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = جھوٹ;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    برائے (صحیح i = 0; ok && i < RDMA_NBUF; i++) {
        اگر (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = جھوٹ;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    اگر (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        واپس جھوٹ;
    }
    اگر (!ok || peer_ready != RDMA_SYNC_READY) {
        واپس جھوٹ;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->راستہ_mtu, RDMA_NBUF);
    واپس سچ;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
صحیح apple_rdma::impl::پیش_رفت() {
    ساخت ibv_wc wc[RDMA_NBUF * 2];
    صحیح n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    اگر (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = سچ; واپس -1; }
    برائے (صحیح j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        منطقی is_recv = (id & RDMA_RECV_WR) != 0;
        اگر (wc[j].حالت_2 != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].حالت_2);
            broken = سچ;
            واپس -1;
        }
        اگر (is_recv) {
            صحیح b = (صحیح)(id & RDMA_WR_IDX_نقاب);
            ثابت rdma_seg_hdr * h = (ثابت rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            اگر (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = سچ; واپس -1; }
            اگر (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = سچ; واپس -1; }
            صحیح خانہ = (inq_سر + inq_گنتی) % RDMA_NBUF;
            inq[خانہ].buf  = b;
            inq[خانہ].off  = 0;
            inq[خانہ].len  = h->len;
            inq_گنتی++;
        } ورنہ {
            send_busy[(صحیح)(id & RDMA_WR_IDX_نقاب)] = 0;
        }
    }
    واپس n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
منطقی apple_rdma::impl::acquire_pending() {
    اگر (pend_buf >= 0) واپس سچ;
    برائے (;;) {
        اگر (broken) واپس جھوٹ;
        برائے (صحیح k = 0; k < RDMA_NBUF; k++) اگر (!send_busy[k]) { pend_buf = k; pend_len = 0; واپس سچ; }
        اگر (پیش_رفت() < 0) واپس جھوٹ;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
منطقی apple_rdma::impl::post_pending() {
    اگر (pend_buf < 0) واپس سچ;
    صحیح i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    اگر (!post_send(i, RDMA_STRIDE)) { broken = سچ; واپس جھوٹ; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    واپس سچ;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
منطقی apple_rdma::send(ثابت خالی * ڈیٹا, size_t size) {
    impl * c = pimpl.get();
    ثابت uint8_t * p = (ثابت uint8_t *)ڈیٹا;
    جب (size > 0) {
        اگر (c->broken) واپس جھوٹ;
        اگر (!c->acquire_pending()) واپس جھوٹ;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        حافظہ_نقل(sb + حجم(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        اگر (c->pend_len == RDMA_PAYLOAD) { اگر (!c->post_pending()) واپس جھوٹ; }
    }
    واپس سچ;
}

منطقی apple_rdma::recv(خالی * ڈیٹا, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)ڈیٹا;
    اگر (!c->post_pending()) واپس جھوٹ;   // turnaround: flush the coalesced request
    غیرعلامتی idle = 0;
    جب (size > 0) {
        اگر (c->inq_گنتی == 0) {
            اگر (c->broken) واپس جھوٹ;
            صحیح n = c->پیش_رفت();
            اگر (n < 0) واپس جھوٹ;
            اگر (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                اگر ((++idle & 0xFFFFF) == 0) {
                    ساخت pollfd pfd = { c->fd, POLLIN, 0 };
                    اگر (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        واپس جھوٹ;
                    }
                }
            } ورنہ {
                idle = 0;
            }
            جاری;
        }
        idle = 0;
        صحیح خانہ = c->inq_سر;
        صحیح b = c->inq[خانہ].buf;
        uint32_t avail = c->inq[خانہ].len - c->inq[خانہ].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        حافظہ_نقل(p, c->recv_mem + (size_t)b * RDMA_STRIDE + حجم(rdma_seg_hdr) + c->inq[خانہ].off, take);
        p += take;
        size -= take;
        c->inq[خانہ].off += take;
        اگر (c->inq[خانہ].off == c->inq[خانہ].len) {
            اگر (!c->post_recv(b)) { c->broken = سچ; واپس جھوٹ; }
            c->inq_سر = (c->inq_سر + 1) % RDMA_NBUF;
            c->inq_گنتی--;
        }
    }
    واپس سچ;
}

منطقی apple_rdma::flush() {
    واپس pimpl->post_pending();
}

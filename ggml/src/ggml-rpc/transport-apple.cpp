#include "transport-apple.h"
#include "transport.h"
#include "log.h"

#include <infiniband/verbs.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dlfcn.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

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
static constexpr uint32_t RDMA_SEG_MAGIC   = 0x52534547u; // "RSEG"
static constexpr целое      RDMA_NBUF        = 16;          // ring depth (frames per direction)
static constexpr size_t   RDMA_FRAME       = 4096;        // Thunderbolt frame (fixed on Apple)
static constexpr size_t   RDMA_STRIDE      = 128 * 1024;  // 32 Thunderbolt frames; NBUF x this = 2 MiB pinned per direction
static constexpr uint32_t RDMA_PSN         = 0;           // any value works if both sides match: UC has no retransmit
static constexpr size_t   RDMA_GID_SIZE    = 16;

static_assert(RDMA_STRIDE % RDMA_FRAME == 0, "RDMA_STRIDE must be a whole number of frames");
// TN3205 counts queue depth in Thunderbolt frames, not work requests.
static constexpr uint32_t RDMA_QP_WR       = (uint32_t)RDMA_NBUF * (RDMA_STRIDE / RDMA_FRAME);
static constexpr uint64_t RDMA_RECV_WR     = 1ull << 20;  // wr_id bit tagging recv completions
static constexpr uint64_t RDMA_WR_IDX_MASK = 0xffff;      // buffer index in the low bits of wr_id
static constexpr uint8_t  RDMA_SYNC_READY  = 0x2A;        // readiness-handshake byte (peer activated)

struct rdma_seg_hdr {
    uint32_t magic; // RDMA_SEG_MAGIC; a mismatch means the stream desynced
    uint32_t len;   // payload bytes in this frame; the rest of the stride is padding
};
static constexpr size_t RDMA_PAYLOAD = RDMA_STRIDE - sizeof(rdma_seg_hdr);

struct apple_rdma_caps {
    uint32_t qpn;
    uint16_t lid;
    uint16_t reserved;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(sizeof(apple_rdma_caps) == RPC_CONN_CAPS_SIZE, "apple_rdma_caps must match conn_caps size");

struct apple_rdma::impl {
    целое fd = -1;                      // bootstrap TCP socket, kept as the liveness anchor

    struct ibv_context * ctx = nullptr;
    struct ibv_pd * pd = nullptr;
    struct ibv_cq * cq = nullptr;     // one CQ for both directions; RDMA_RECV_WR tags recv completions
    struct ibv_qp * qp = nullptr;

    uint8_t       * send_mem = nullptr;
    struct ibv_mr * send_mr  = nullptr;
    uint8_t       * recv_mem = nullptr;
    struct ibv_mr * recv_mr  = nullptr;

    целое      send_busy[RDMA_NBUF] = {};  // 1 while this buffer has a send in flight
    // completed recv frames, oldest first: ring index, bytes already handed to
    // the reader, and total payload length
    struct { целое buf; uint32_t off; uint32_t len; } inq[RDMA_NBUF] = {};
    целое      inq_head = 0;
    целое      inq_count = 0;
    целое      pend_buf = -1;
    uint32_t pend_len = 0;
    bool     broken = false;

    uint32_t     qpn = 0;
    uint8_t      port = 0;
    целое          gid_idx = 0;
    enum ibv_mtu path_mtu = IBV_MTU_1024;

    целое  progress();
    bool acquire_pending();
    bool post_pending();

    bool post_recv(целое i) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)(recv_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)RDMA_STRIDE;
        sge.lkey   = recv_mr->lkey;
        struct ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = RDMA_RECV_WR | (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        вернуть ibv_post_recv(qp, &wr, &bad) == 0;
    }

    bool post_send(целое i, size_t len) {
        struct ibv_sge sge = {};
        sge.addr   = (uintptr_t)(send_mem + (size_t)i * RDMA_STRIDE);
        sge.length = (uint32_t)len;
        sge.lkey   = send_mr->lkey;
        struct ibv_send_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode  = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;
        вернуть ibv_post_send(qp, &wr, &bad) == 0;
    }

    ~impl() {
        broken = true;
        // destroy the QP first: it can still write to the rings until it is gone.
        // no IBV_QPS_ERR before it - Apple's provider then fails every region unmap.
        если (qp) ibv_destroy_qp(qp);
        если (send_mr) ibv_dereg_mr(send_mr);
        если (recv_mr) ibv_dereg_mr(recv_mr);
        free(send_mem);
        free(recv_mem);
        если (cq)  ibv_destroy_cq(cq);
        если (pd)  ibv_dealloc_pd(pd);
        если (ctx) ibv_close_device(ctx);
    }
};

apple_rdma::apple_rdma(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

apple_rdma::~apple_rdma() = default;

bool apple_rdma::broken() const {
    вернуть pimpl->broken;
}

// The readiness handshake below still runs over the bootstrap socket, one byte
// each way, before the transport is declared live.
static bool tcp_send_byte(целое fd, uint8_t b) {
    ssize_t n;
    do { n = ::send(fd, &b, sizeof(b), 0); } пока (n < 0 && errno == EINTR);
    вернуть n == sizeof(b);
}

static bool tcp_recv_byte(целое fd, uint8_t * b) {
    ssize_t n;
    do { n = ::recv(fd, b, sizeof(*b), 0); } пока (n < 0 && errno == EINTR);
    вернуть n == (ssize_t)sizeof(*b);
}

// Index of the GID on this port equal to the target, or -1. Thunderbolt GIDs are
// RoCEv2 IPv4-mapped (::ffff:a.b.c.d), so this matches the local TCP address.
static целое rdma_match_gid(struct ibv_context * ctx, uint8_t port, целое gid_tbl_len,
                          const uint8_t * target, union ibv_gid * out) {
    для (целое i = 0; i < gid_tbl_len; i++) {
        union ibv_gid g;
        если (ibv_query_gid(ctx, port, i, &g) != 0) continue;
        если (memcmp(g.raw, target, RDMA_GID_SIZE) != 0) continue;
        если (out) *out = g;
        вернуть i;
    }
    вернуть -1;
}

// First ACTIVE port on the device. Only a cabled, up Thunderbolt link reports
// ACTIVE, and it is not always port 1, so the port cannot be hardcoded the way
// the Linux path does. Returns 0 if none.
static uint8_t rdma_first_active_port(struct ibv_context * ctx, struct ibv_port_attr * out) {
    struct ibv_device_attr da;
    если (ibv_query_device(ctx, &da) != 0) вернуть 0;
    для (uint8_t p = 1; p <= da.phys_port_cnt; p++) {
        struct ibv_port_attr pa;
        если (ibv_query_port(ctx, p, &pa) != 0) continue;
        если (pa.state == IBV_PORT_ACTIVE) { если (out) *out = pa; вернуть p; }
    }
    вернуть 0;
}

// librdma.dylib is weak-linked, so its symbols are null when it is absent. Nothing may
// call one before this has returned true.
static bool rdma_library_present() {
    static const bool present = [] {
        void * handle = dlopen("/usr/lib/librdma.dylib", RTLD_LAZY);
        если (handle == nullptr) {
            вернуть false;
        }
        dlclose(handle);
        вернуть true;
    }();
    вернуть present;
}

// Called before the endpoints are exchanged: pick the local device facing this
// peer, create a UC QP and register the frame rings. RDMA is point-to-point, so
// the device is the one whose GID equals the bootstrap connection's local
// address, i.e. the one cabled to the peer.
std::unique_ptr<apple_rdma> apple_rdma::probe(целое fd, const uint8_t * target_gid, uint8_t * caps) {
    если (!rdma_library_present()) {
        LOG_DBG2("[RDMA(Apple)] librdma.dylib not present, continuing with TCP\n");
        вернуть nullptr;
    }
    целое ndev = 0;
    ibv_device ** devs = ibv_get_device_list(&ndev);
    если (!devs) вернуть nullptr;

    ibv_context * ctx = nullptr;
    uint8_t port = 0;
    struct ibv_port_attr pa = {};
    union ibv_gid gid = {};
    целое gid_idx = -1;
    std::string matched;
    для (целое d = 0; d < ndev; d++) {
        ibv_context * c = ibv_open_device(devs[d]);
        если (!c) continue;
        struct ibv_port_attr p = {};
        uint8_t pt = rdma_first_active_port(c, &p);
        целое gi = pt ? rdma_match_gid(c, pt, p.gid_tbl_len, target_gid, &gid) : -1;
        если (gi < 0) { ibv_close_device(c); continue; }
        ctx = c; port = pt; pa = p; gid_idx = gi;
        const char * name = ibv_get_device_name(devs[d]);
        matched = name ? name : "";
        break;
    }
    ibv_free_device_list(devs);
    если (!ctx) {
        LOG_DBG2("[RDMA(Apple)] no RDMA device matched the connection address, continuing with TCP\n");
        вернуть nullptr;
    }

    std::unique_ptr<impl> c(new impl());
    c->fd  = fd;
    c->ctx = ctx;
    c->port = port;
    c->gid_idx = gid_idx;
    c->path_mtu = pa.active_mtu;

    c->pd = ibv_alloc_pd(ctx);
    если (!c->pd) вернуть nullptr;

    c->cq = ibv_create_cq(ctx, 2 * RDMA_QP_WR + 1, nullptr, nullptr, 0);
    если (!c->cq) вернуть nullptr;

    ibv_qp_init_attr qia = {};
    qia.send_cq = c->cq;
    qia.recv_cq = c->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr  = RDMA_QP_WR;
    qia.cap.max_recv_wr  = RDMA_QP_WR;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    c->qp = ibv_create_qp(c->pd, &qia);
    если (!c->qp) вернуть nullptr;

    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
        если (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            вернуть nullptr;
        }
    }

    long page = sysconf(_SC_PAGESIZE);
    если (page <= 0) page = 4096;
    const size_t ring_bytes = (size_t)RDMA_NBUF * RDMA_STRIDE;
    если (posix_memalign((void **)&c->send_mem, (size_t)page, ring_bytes) != 0) c->send_mem = nullptr;
    если (posix_memalign((void **)&c->recv_mem, (size_t)page, ring_bytes) != 0) c->recv_mem = nullptr;
    если (!c->send_mem || !c->recv_mem) вернуть nullptr;

    // Apple's provider rejects LOCAL_WRITE-only MRs even for two-sided SEND/RECV.
    const целое mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    c->send_mr = ibv_reg_mr(c->pd, c->send_mem, ring_bytes, mr_flags);
    c->recv_mr = ibv_reg_mr(c->pd, c->recv_mem, ring_bytes, mr_flags);
    если (!c->send_mr || !c->recv_mr) вернуть nullptr;

    // Recvs are posted in activate() after the RTS transition, not here: Apple's
    // provider rejects ibv_post_recv on a QP that has not reached RTS.

    c->qpn = c->qp->qp_num;

    apple_rdma_caps rc = {};
    rc.qpn = c->qpn;
    rc.lid = pa.lid;
    memcpy(rc.gid, gid.raw, RDMA_GID_SIZE);
    memcpy(caps, &rc, sizeof(rc));

    LOG_INFO("RDMA(Apple/UC) probed: dev=%s port=%u gid=%d qpn=%u lid=%u mtu=%d ring=%d x %zu KiB\n",
                  matched.c_str(), port, gid_idx, c->qpn, (unsigned)pa.lid, 128 << c->path_mtu,
                  RDMA_NBUF, RDMA_STRIDE / 1024);
    вернуть std::unique_ptr<apple_rdma>(new apple_rdma(std::move(c)));
}

// Called once the peer's endpoint has arrived: INIT -> RTR -> RTS (UC: GID/GRH
// addressing, no timeout/retry/rnr/rd_atomic), then the readiness handshake.
bool apple_rdma::activate(const uint8_t * caps) {
    impl * c = pimpl.get();

    apple_rdma_caps rc = {};
    memcpy(&rc, caps, sizeof(rc));

    bool ok = true;
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
        если (ibv_modify_qp(c->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTR failed: %s\n", strerror(errno));
            ok = false;
        }
    }
    если (ok) {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.sq_psn   = RDMA_PSN;
        если (ibv_modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
            LOG_ERROR("RDMA(Apple/UC) RTS failed: %s\n", strerror(errno));
            ok = false;
        }
    }

    // Recvs are posted only now: the controller starts processing them at RTR.
    для (целое i = 0; ok && i < RDMA_NBUF; i++) {
        если (!c->post_recv(i)) {
            LOG_ERROR("RDMA(Apple/UC) post_recv %d/%d failed\n", i, RDMA_NBUF);
            ok = false;
        }
    }

    // A queue pair processes receives only after RTR and the transitions above can
    // fail on one side alone, so neither peer sends a frame until both report their
    // recvs posted.
    uint8_t peer_ready = 0;
    если (!tcp_send_byte(c->fd, ok ? RDMA_SYNC_READY : 0) || !tcp_recv_byte(c->fd, &peer_ready)) {
        вернуть false;
    }
    если (!ok || peer_ready != RDMA_SYNC_READY) {
        вернуть false;
    }

    LOG_INFO("RDMA(Apple/UC) activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  c->qpn, rc.qpn, 128 << c->path_mtu, RDMA_NBUF);
    вернуть true;
}

// Drain the CQ: release completed send buffers, queue completed recv frames for
// the reader. Returns the number of completions reaped, or -1 on error.
целое apple_rdma::impl::progress() {
    struct ibv_wc wc[RDMA_NBUF * 2];
    целое n = ibv_poll_cq(cq, RDMA_NBUF * 2, wc);
    если (n < 0) { LOG_ERROR("RDMA(Apple/UC) poll_cq failed\n"); broken = true; вернуть -1; }
    для (целое j = 0; j < n; j++) {
        uint64_t id = wc[j].wr_id;
        bool is_recv = (id & RDMA_RECV_WR) != 0;
        если (wc[j].status != IBV_WC_SUCCESS) {
            LOG_ERROR("RDMA(Apple/UC) %s wc error: status=%d\n", is_recv ? "recv" : "send", wc[j].status);
            broken = true;
            вернуть -1;
        }
        если (is_recv) {
            целое b = (целое)(id & RDMA_WR_IDX_MASK);
            const rdma_seg_hdr * h = (const rdma_seg_hdr *)(recv_mem + (size_t)b * RDMA_STRIDE);
            если (h->magic != RDMA_SEG_MAGIC) { LOG_ERROR("RDMA(Apple/UC) bad frame magic\n"); broken = true; вернуть -1; }
            если (h->len > RDMA_PAYLOAD) { LOG_ERROR("RDMA(Apple/UC) frame len %u exceeds payload\n", h->len); broken = true; вернуть -1; }
            целое slot = (inq_head + inq_count) % RDMA_NBUF;
            inq[slot].buf  = b;
            inq[slot].off  = 0;
            inq[slot].len  = h->len;
            inq_count++;
        } иначе {
            send_busy[(целое)(id & RDMA_WR_IDX_MASK)] = 0;
        }
    }
    вернуть n;
}

// Reserve a free send buffer to coalesce into, waiting on progress if none free.
bool apple_rdma::impl::acquire_pending() {
    если (pend_buf >= 0) вернуть true;
    для (;;) {
        если (broken) вернуть false;
        для (целое k = 0; k < RDMA_NBUF; k++) если (!send_busy[k]) { pend_buf = k; pend_len = 0; вернуть true; }
        если (progress() < 0) вернуть false;
    }
}

// Post the pending frame. The whole STRIDE goes out even when only partly filled:
// TN3205 requires a SEND and its matching RECV to cover the same number of
// Thunderbolt frames, so a short send would fail the peer's receive.
bool apple_rdma::impl::post_pending() {
    если (pend_buf < 0) вернуть true;
    целое i = pend_buf;
    rdma_seg_hdr * h = (rdma_seg_hdr *)(send_mem + (size_t)i * RDMA_STRIDE);
    h->magic = RDMA_SEG_MAGIC;
    h->len   = pend_len;
    если (!post_send(i, RDMA_STRIDE)) { broken = true; вернуть false; }
    send_busy[i] = 1;
    pend_buf = -1;
    pend_len = 0;
    вернуть true;
}

// Coalescing write: append into the pending frame, posting a full frame when it
// fills. The trailing partial is posted by flush() at each message boundary.
bool apple_rdma::send(const void * data, size_t size) {
    impl * c = pimpl.get();
    const uint8_t * p = (const uint8_t *)data;
    пока (size > 0) {
        если (c->broken) вернуть false;
        если (!c->acquire_pending()) вернуть false;
        uint8_t * sb = c->send_mem + (size_t)c->pend_buf * RDMA_STRIDE;
        size_t space = RDMA_PAYLOAD - c->pend_len;
        size_t chunk = size < space ? size : space;
        memcpy(sb + sizeof(rdma_seg_hdr) + c->pend_len, p, chunk);
        c->pend_len += (uint32_t)chunk;
        p += chunk;
        size -= chunk;
        если (c->pend_len == RDMA_PAYLOAD) { если (!c->post_pending()) вернуть false; }
    }
    вернуть true;
}

bool apple_rdma::recv(void * data, size_t size) {
    impl * c = pimpl.get();
    uint8_t * p = (uint8_t *)data;
    если (!c->post_pending()) вернуть false;   // turnaround: flush the coalesced request
    unsigned idle = 0;
    пока (size > 0) {
        если (c->inq_count == 0) {
            если (c->broken) вернуть false;
            целое n = c->progress();
            если (n < 0) вернуть false;
            если (n == 0) {
                // UC gives no disconnect notification, so the bootstrap TCP fd is
                // the liveness anchor: nothing crosses it once RDMA is up, so any
                // readability means the peer's FIN (macOS has no POLLRDHUP).
                // Same idle interval as the Linux path.
                если ((++idle & 0xFFFFF) == 0) {
                    struct pollfd pfd = { c->fd, POLLIN, 0 };
                    если (poll(&pfd, 1, 0) > 0 &&
                        (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))) {
                        вернуть false;
                    }
                }
            } иначе {
                idle = 0;
            }
            continue;
        }
        idle = 0;
        целое slot = c->inq_head;
        целое b = c->inq[slot].buf;
        uint32_t avail = c->inq[slot].len - c->inq[slot].off;
        uint32_t take = (size < (size_t)avail) ? (uint32_t)size : avail;
        memcpy(p, c->recv_mem + (size_t)b * RDMA_STRIDE + sizeof(rdma_seg_hdr) + c->inq[slot].off, take);
        p += take;
        size -= take;
        c->inq[slot].off += take;
        если (c->inq[slot].off == c->inq[slot].len) {
            если (!c->post_recv(b)) { c->broken = true; вернуть false; }
            c->inq_head = (c->inq_head + 1) % RDMA_NBUF;
            c->inq_count--;
        }
    }
    вернуть true;
}

bool apple_rdma::flush() {
    вернуть pimpl->post_pending();
}

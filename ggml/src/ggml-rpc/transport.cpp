#include "transport.h"
#include "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#បើមិនដូច្នេះ
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>

#ifdef GGML_RPC_RDMA
#  include <infiniband/verbs.h>
#  include <array>
#  include <cerrno>
#  include <chrono>
#  include <time.h>
#  ifndef _WIN32
#    include <poll.h>
#  endif
#  ifdef GGML_RPC_RDMA_APPLE
#    include "transport-apple.h"
#  endif
#endif // GGML_RPC_RDMA

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#បើមិនដូច្នេះ
typedef ចំនួនគត់ sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
static constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#ប្រសិនបើ defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
static constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
static constexpr ចំនួនគត់    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
static constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

sa_nracanaa rdma_conn {
    sa_nracanaa ibv_context * ctx = nullptr;
    sa_nracanaa ibv_pd * pd  = nullptr;
    sa_nracanaa ibv_cq * scq = nullptr;   // send completions
    sa_nracanaa ibv_cq * rcq = nullptr;   // recv completions
    sa_nracanaa ibv_qp * qp  = nullptr;
    sa_nracanaa ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    void          * tx_buf = nullptr;
    sa_nracanaa ibv_mr * tx_mr  = nullptr;

    void          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    sa_nracanaa ibv_mr * rx_mr  = nullptr;
    ចំនួនគត់             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(ចំនួនគត់ i) const {
        ត្រឡប់ static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    bool post_rx(ចំនួនគត់ i) {
        sa_nracanaa ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        sa_nracanaa ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        ត្រឡប់ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        ប្រសិនបើ (tx_mr) ibv_dereg_mr(tx_mr);
        ប្រសិនបើ (rx_mr) ibv_dereg_mr(rx_mr);
        free(tx_buf);
        free(rx_buf);
        ប្រសិនបើ (qp)  ibv_destroy_qp(qp);
        ប្រសិនបើ (scq) ibv_destroy_cq(scq);
        ប្រសិនបើ (rcq) ibv_destroy_cq(rcq);
        ប្រសិនបើ (ch)  ibv_destroy_comp_channel(ch);
        ប្រសិនបើ (pd)  ibv_dealloc_pd(pd);
        ប្រសិនបើ (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
sa_nracanaa rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    ចំនួនគត់      gid_idx = 0;
    ga_nanaa ibv_mtu path_mtu = IBV_MTU_1024;
};

sa_nracanaa rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(parimaa_na(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

sa_nracanaa socket_t::impl {
    impl(sockfd_t fd) : use_rdma(false), fd(fd) {}
    ~impl();
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    bool flush();
    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  បើមិនដូច្នេះ
    bool rdma_probe();
    bool rdma_send(const void * data, size_t size);
    bool rdma_recv(void * data, size_t size);
    bool tcp_peer_closed();
    bool rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid);
    bool rdma_poll(sa_nracanaa ibv_cq * cq, sa_nracanaa ibv_wc * wc);
    bool rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    bool     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
    ប្រសិនបើ (fd != INVALID_SOCKET) closesocket(this->fd);
#បើមិនដូច្នេះ
    ប្រសិនបើ (fd >= 0) close(this->fd);
#endif
}

#ifdef GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_target_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = parimaa_na(addr);
    ប្រសិនបើ (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        ត្រឡប់ std::nullopt;
    }
    rdma_gid_t target = {};
    ប្រសិនបើ (addr.ss_family == AF_INET) {
        const auto * a = reinterpret_cast<const sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        ត្រឡប់ target;
    }
    ប្រសិនបើ (addr.ss_family == AF_INET6) {
        const auto * a = reinterpret_cast<const sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        ត្រឡប់ target;
    }
    ត្រឡប់ std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

bool socket_t::impl::tcp_peer_closed() {
    ប្រសិនបើ (fd < 0) ត្រឡប់ false;
#ifndef _WIN32
    sa_nracanaa pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    ចំនួនគត់ r = poll(&pfd, 1, 0);
    ត្រឡប់ r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#បើមិនដូច្នេះ
    ត្រឡប់ false;
#endif
}

bool socket_t::impl::rdma_probe() {
    const var_na * dev_env = std::getenv("GGML_RDMA_DEV");
    const var_na * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    ប្រសិនបើ (!target_gid) {
        ត្រឡប់ false;
    }

    const uint8_t ib_port = 1;
    ចំនួនគត់ num_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&num_devs);
    ប្រសិនបើ (!devs || num_devs == 0) ត្រឡប់ false;

    ibv_context * ibctx = nullptr;
    const var_na * matched_dev = nullptr;
    ចំនួនគត់ gid_idx = gid_env ? atoi(gid_env) : -1;
    ចំនួនគត់ gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    សម្រាប់ (ចំនួនគត់ d = 0; d < num_devs; d++) {
        const var_na * dn = ibv_get_device_name(devs[d]);
        ប្រសិនបើ (dev_env && strcmp(dev_env, dn) != 0) continue;

        ibv_context * ctx = ibv_open_device(devs[d]);
        ប្រសិនបើ (!ctx) continue;

        ibv_port_attr pa;
        ប្រសិនបើ (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); continue; }

        ចំនួនគត់ found_gid = gid_idx;
        ចំនួនគត់ found_version = IBV_GID_TYPE_IB;
        ប្រសិនបើ (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            ចំនួនគត់ v2_idx = -1;
            ចំនួនគត់ v1_idx = -1;
            សម្រាប់ (ចំនួនគត់ i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                ប្រសិនបើ (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) continue;
                ប្រសិនបើ (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) continue;
                ប្រសិនបើ (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } បើមិនដូច្នេះ ប្រសិនបើ (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            ប្រសិនបើ (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } បើមិនដូច្នេះ ប្រសិនបើ (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } បើមិនដូច្នេះ {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            ប្រសិនបើ (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        ប្រសិនបើ (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            bha_nga;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    ប្រសិនបើ (!ibctx) ត្រឡប់ false;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    ប្រសិនបើ (!rdma->pd) ត្រឡប់ false;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    ប្រសិនបើ (!rdma->scq || !rdma->rcq) ត្រឡប់ false;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_DEPTH + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_data = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    ប្រសិនបើ (!rdma->qp) ត្រឡប់ false;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    ប្រសិនបើ (!rdma->tx_buf || !rdma->rx_buf) ត្រឡប់ false;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ប្រសិនបើ (!rdma->tx_mr || !rdma->rx_mr) ត្រឡប់ false;

    ibv_gid local_gid;
    ប្រសិនបើ (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) ត្រឡប់ false;

    rdma_local.qpn = rdma->qp->qp_num;
    rdma_local.psn = rdma->qp->qp_num & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    const var_na * ver_str = "";
    ប្រសិនបើ (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } បើមិនដូច្នេះ ប្រសិនបើ (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    ត្រឡប់ true;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
bool socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, const uint8_t * remote_gid) {
    // RESET -> INIT
    {
        sa_nracanaa ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_num        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        ប្រសិនបើ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            ត្រឡប់ false;
        }
    }

    សម្រាប់ (ចំនួនគត់ i = 0; i < RDMA_RX_DEPTH; i++) {
        ប្រសិនបើ (!rdma->post_rx(i)) ត្រឡប់ false;
    }

    // INIT -> RTR
    {
        sa_nracanaa ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_num        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_num   = rdma_local.ib_port;
        ប្រសិនបើ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            ត្រឡប់ false;
        }
    }

    // RTR -> RTS
    {
        sa_nracanaa ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        ប្រសិនបើ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            ត្រឡប់ false;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    ត្រឡប់ true;
}

// Sleep until the completion channel has an event or the TCP peer closes.
bool socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    sa_nracanaa pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    ប្រសិនបើ (poll(pfds, 2, -1) < 0) {
        ត្រឡប់ errno == EINTR;
    }
    ប្រសិនបើ (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        ត្រឡប់ false;
    }
    ប្រសិនបើ (pfds[0].revents & POLLIN) {
        sa_nracanaa ibv_cq * ev_cq  = nullptr;
        void          * ev_ctx = nullptr;
        ប្រសិនបើ (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            ត្រឡប់ false;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    ត្រឡប់ true;
}

bool socket_t::impl::rdma_poll(sa_nracanaa ibv_cq * cq, sa_nracanaa ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    bool armed = false;
    សម្រាប់ (uint64_t s = 1; ; s++) {
        ចំនួនគត់ n = ibv_poll_cq(cq, 1, wc);
        ប្រសិនបើ (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            ប្រសិនបើ (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            ត្រឡប់ wc->status == IBV_WC_SUCCESS;
        }
        ប្រសិនបើ (n < 0) ត្រឡប់ false;
        ប្រសិនបើ (armed) {
            // armed and still empty: sleep until the next completion
            ប្រសិនបើ (!rdma_wait_event()) {
                ត្រឡប់ false;
            }
            armed = false;
            continue;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        ប្រសិនបើ (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            ប្រសិនបើ (ibv_req_notify_cq(cq, 0) != 0) {
                ត្រឡប់ false;
            }
            armed = true;
            continue;
        }
        ប្រសិនបើ ((s & 0xFFFFF) == 0) {
            ប្រសិនបើ (tcp_peer_closed()) {
                ត្រឡប់ false;
            }
        }
    }
}

bool socket_t::impl::rdma_send(const void * data, size_t size) {
    rdma_conn * c = rdma.get();
    const uint8_t * src = (const uint8_t *)data;
    size_t rem = size;
    ខណៈពេល (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        sa_nracanaa ibv_sge sge = {};
        sa_nracanaa ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        ប្រសិនបើ (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } បើមិនដូច្នេះ {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        ប្រសិនបើ (ibv_post_send(c->qp, &wr, &bad) != 0) ត្រឡប់ false;
        c->last_active = std::chrono::steady_clock::now();
        sa_nracanaa ibv_wc wc;
        ប្រសិនបើ (!rdma_poll(c->scq, &wc)) ត្រឡប់ false;

        src += chunk;
        rem -= chunk;
    }
    ត្រឡប់ true;
}

bool socket_t::impl::rdma_recv(void * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    ខណៈពេល (rem > 0) {
        sa_nracanaa ibv_wc wc;
        ប្រសិនបើ (!rdma_poll(c->rcq, &wc)) ត្រឡប់ false;

        ចំនួនគត់ slot = (ចំនួនគត់)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        ប្រសិនបើ (!c->post_rx(slot)) ត្រឡប់ false;

        dst += got;
        rem -= got;
    }
    ត្រឡប់ true;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

bool socket_t::impl::send_data(const void * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    ប្រសិនបើ (use_rdma) {
        ត្រឡប់ rdma->send(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    ប្រសិនបើ (use_rdma) {
        ត្រឡប់ rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    ខណៈពេល (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (const var_na *)data + bytes_sent, size_to_send, 0);
        ប្រសិនបើ (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            ត្រឡប់ false;
        }
        bytes_sent += (size_t)n;
    }
    ត្រឡប់ true;
}

bool socket_t::impl::recv_data(void * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    ប្រសិនបើ (use_rdma) {
        ត្រឡប់ rdma->recv(data, size);
    }
#elif defined(GGML_RPC_RDMA)
    ប្រសិនបើ (use_rdma) {
        ត្រឡប់ rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    ខណៈពេល (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (var_na *)data + bytes_recv, size_to_recv, 0);
        ប្រសិនបើ (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            ត្រឡប់ false;
        }
        ប្រសិនបើ (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            ត្រឡប់ false;
        }
        bytes_recv += (size_t)n;
    }
    ត្រឡប់ true;
}

void socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    ប្រសិនបើ (std::getenv("GGML_RPC_NO_RDMA")) {
        ត្រឡប់;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    ប្រសិនបើ (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    ប្រសិនបើ (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  បើមិនដូច្នេះ
    rdma_local = {};
    ប្រសិនបើ (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, parimaa_na(rc));
    } បើមិនដូច្នេះ {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

void socket_t::impl::update_caps(const uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    bool remote_rdma = false;
    សម្រាប់ (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    ប្រសិនបើ (!rdma || !remote_rdma) {
        ប្រសិនបើ (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        ត្រឡប់;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    bool activated = rdma->activate(remote_caps);
#  បើមិនដូច្នេះ
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, parimaa_na(rc));
    bool activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    ប្រសិនបើ (activated) {
        use_rdma = true;
    } បើមិនដូច្នេះ {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#បើមិនដូច្នេះ
    (void)remote_caps;
#endif // GGML_RPC_RDMA
}

bool socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    ប្រសិនបើ (use_rdma) {
        ត្រឡប់ rdma->flush();
    }
#endif
    ត្រឡប់ true;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = default;

bool socket_t::send_data(const void * data, size_t size) {
    ត្រឡប់ pimpl->send_data(data, size);
}

bool socket_t::recv_data(void * data, size_t size) {
    ត្រឡប់ pimpl->recv_data(data, size);
}

bool socket_t::flush() {
    ត្រឡប់ pimpl->flush();
}

void socket_t::get_caps(uint8_t * local_caps) {
    ត្រឡប់ pimpl->get_caps(local_caps);
}

void socket_t::update_caps(const uint8_t * remote_caps) {
    ត្រឡប់ pimpl->update_caps(remote_caps);
}

const var_na * socket_t::transport_name() const {
    ត្រឡប់ pimpl->use_rdma ? "RDMA" : "TCP";
}

static bool is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    ត្រឡប់ sockfd != INVALID_SOCKET;
#បើមិនដូច្នេះ
    ត្រឡប់ sockfd >= 0;
#endif
}

static bool set_no_delay(sockfd_t sockfd) {
    ចំនួនគត់ flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    ចំនួនគត់ ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (var_na *)&flag, parimaa_na(ចំនួនគត់));
    ត្រឡប់ ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    ចំនួនគត់ flag = 1;
    ចំនួនគត់ ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (var_na *)&flag, parimaa_na(ចំនួនគត់));
    ត្រឡប់ ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, NULL, NULL);
    ប្រសិនបើ (!is_valid_fd(client_socket_fd)) {
        ត្រឡប់ nullptr;
    }
    ប្រសិនបើ (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ត្រឡប់ nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    ត្រឡប់ socket_ptr(new socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(const var_na * host, ចំនួនគត់ port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    ប្រសិនបើ (!is_valid_fd(sockfd)) {
        ត្រឡប់ nullptr;
    }
    ប្រសិនបើ (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        ត្រឡប់ nullptr;
    }
    ប្រសិនបើ (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        ត្រឡប់ nullptr;
    }
    sa_nracanaa sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    ប្រសិនបើ (bind(sockfd, (sa_nracanaa sockaddr *) &serv_addr, parimaa_na(serv_addr)) < 0) {
        ត្រឡប់ nullptr;
    }
    ប្រសិនបើ (listen(sockfd, 1) < 0) {
        ត្រឡប់ nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    ត្រឡប់ socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(const var_na * host, ចំនួនគត់ port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    ប្រសិនបើ (!is_valid_fd(sockfd)) {
        ត្រឡប់ nullptr;
    }
    ប្រសិនបើ (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ត្រឡប់ nullptr;
    }
    sa_nracanaa sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    sa_nracanaa hostent * server = gethostbyname(host);
    ប្រសិនបើ (server == NULL) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        ត្រឡប់ nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    ប្រសិនបើ (::connect(sockfd, (sa_nracanaa sockaddr *)&addr, parimaa_na(addr)) < 0) {
        ត្រឡប់ nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    ត្រឡប់ socket_ptr(new socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
static std::mutex g_rpc_transport_mu;
static bool g_rpc_transport_wsa_started = false;
#endif

bool rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    ប្រសិនបើ (g_rpc_transport_wsa_started) {
        ត្រឡប់ true;
    }
    WSADATA wsaData;
    ចំនួនគត់ res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    ប្រសិនបើ (res != 0) {
        ត្រឡប់ false;
    }
    g_rpc_transport_wsa_started = true;
    ត្រឡប់ true;
#បើមិនដូច្នេះ
    ត្រឡប់ true;
#endif
}

void rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    ប្រសិនបើ (!g_rpc_transport_wsa_started) {
        ត្រឡប់;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = false;
#endif
}

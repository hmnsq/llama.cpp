#ଅନ୍ତର୍ଭାଵ "transport.h"
#ଅନ୍ତର୍ଭାଵ "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#ଅନ୍ୟଥା
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <mutex>
#ଅନ୍ତର୍ଭାଵ <optional>

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
ପ୍ରକାରବ୍ୟାଖ୍ୟା SOCKET sockfd_t;
using ssize_t = __int64;
#ଅନ୍ୟଥା
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
ସ୍ଥିର constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#ଯଦି defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ସ୍ଥିର constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ସ୍ଥିର constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

ସଂରଚନା rdma_conn {
    ସଂରଚନା ibv_context * ctx = nullptr;
    ସଂରଚନା ibv_pd * pd  = nullptr;
    ସଂରଚନା ibv_cq * scq = nullptr;   // send completions
    ସଂରଚନା ibv_cq * rcq = nullptr;   // recv completions
    ସଂରଚନା ibv_qp * qp  = nullptr;
    ସଂରଚନା ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    ଶୂନ୍ୟ          * tx_buf = nullptr;
    ସଂରଚନା ibv_mr * tx_mr  = nullptr;

    ଶୂନ୍ୟ          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    ସଂରଚନା ibv_mr * rx_mr  = nullptr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) ଅଚଳ {
        ଫେରନ୍ତୁ static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    ତାର୍କିକ post_rx(ପୂର୍ଣ୍ଣସଂଖ୍ୟା i) {
        ସଂରଚନା ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        ସଂରଚନା ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ଅଙ୍କ_sge = 1;
        ଫେରନ୍ତୁ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        ଯଦି (tx_mr) ibv_dereg_mr(tx_mr);
        ଯଦି (rx_mr) ibv_dereg_mr(rx_mr);
        ମୁକ୍ତ(tx_buf);
        ମୁକ୍ତ(rx_buf);
        ଯଦି (qp)  ibv_destroy_qp(qp);
        ଯଦି (scq) ibv_destroy_cq(scq);
        ଯଦି (rcq) ibv_destroy_cq(rcq);
        ଯଦି (ch)  ibv_destroy_comp_channel(ch);
        ଯଦି (pd)  ibv_dealloc_pd(pd);
        ଯଦି (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
ସଂରଚନା rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା      gid_idx = 0;
    ଗଣନା ibv_mtu path_mtu = IBV_MTU_1024;
};

ସଂରଚନା rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(ଆକାର(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

ସଂରଚନା socket_t::impl {
    impl(sockfd_t fd) : use_rdma(ମିଥ୍ୟା), fd(fd) {}
    ~impl();
    ତାର୍କିକ send_data(ଅଚଳ ଶୂନ୍ୟ * data, size_t size);
    ତାର୍କିକ recv_data(ଶୂନ୍ୟ * data, size_t size);
    ତାର୍କିକ flush();
    ଶୂନ୍ୟ get_caps(uint8_t * local_caps);
    ଶୂନ୍ୟ update_caps(ଅଚଳ uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  ଅନ୍ୟଥା
    ତାର୍କିକ rdma_probe();
    ତାର୍କିକ rdma_send(ଅଚଳ ଶୂନ୍ୟ * data, size_t size);
    ତାର୍କିକ rdma_recv(ଶୂନ୍ୟ * data, size_t size);
    ତାର୍କିକ tcp_peer_closed();
    ତାର୍କିକ rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ଅଚଳ uint8_t * remote_gid);
    ତାର୍କିକ rdma_poll(ସଂରଚନା ibv_cq * cq, ସଂରଚନା ibv_wc * wc);
    ତାର୍କିକ rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    ତାର୍କିକ     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, ଏହି->fd);
#ifdef _WIN32
    ଯଦି (fd != INVALID_SOCKET) closesocket(ଏହି->fd);
#ଅନ୍ୟଥା
    ଯଦି (fd >= 0) close(ଏହି->fd);
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
    socklen_t addr_len = ଆକାର(addr);
    ଯଦି (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        ଫେରନ୍ତୁ std::nullopt;
    }
    rdma_gid_t target = {};
    ଯଦି (addr.ss_family == AF_INET) {
        ଅଚଳ auto * a = reinterpret_cast<ଅଚଳ sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        ଫେରନ୍ତୁ target;
    }
    ଯଦି (addr.ss_family == AF_INET6) {
        ଅଚଳ auto * a = reinterpret_cast<ଅଚଳ sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        ଫେରନ୍ତୁ target;
    }
    ଫେରନ୍ତୁ std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

ତାର୍କିକ socket_t::impl::tcp_peer_closed() {
    ଯଦି (fd < 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
#ifndef _WIN32
    ସଂରଚନା pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = poll(&pfd, 1, 0);
    ଫେରନ୍ତୁ r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
#endif
}

ତାର୍କିକ socket_t::impl::rdma_probe() {
    ଅଚଳ ଅକ୍ଷର * dev_env = std::getenv("GGML_RDMA_DEV");
    ଅଚଳ ଅକ୍ଷର * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    ଯଦି (!target_gid) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ uint8_t ib_port = 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&ଅଙ୍କ_devs);
    ଯଦି (!devs || ଅଙ୍କ_devs == 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ibv_context * ibctx = nullptr;
    ଅଚଳ ଅକ୍ଷର * matched_dev = nullptr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା gid_idx = gid_env ? atoi(gid_env) : -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 0; d < ଅଙ୍କ_devs; d++) {
        ଅଚଳ ଅକ୍ଷର * dn = ibv_get_device_name(devs[d]);
        ଯଦି (dev_env && strcmp(dev_env, dn) != 0) ଜାରି;

        ibv_context * ctx = ibv_open_device(devs[d]);
        ଯଦି (!ctx) ଜାରି;

        ibv_port_attr pa;
        ଯଦି (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); ଜାରି; }

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା found_gid = gid_idx;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା found_version = IBV_GID_TYPE_IB;
        ଯଦି (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା v2_idx = -1;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା v1_idx = -1;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                ଯଦି (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) ଜାରି;
                ଯଦି (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) ଜାରି;
                ଯଦି (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } ଅନ୍ୟଥା ଯଦି (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            ଯଦି (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } ଅନ୍ୟଥା ଯଦି (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } ଅନ୍ୟଥା {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            ଯଦି (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        ଯଦି (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            ଭାଙ୍ଗ;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    ଯଦି (!ibctx) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    ଯଦି (!rdma->pd) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    ଯଦି (!rdma->scq || !rdma->rcq) ଫେରନ୍ତୁ ମିଥ୍ୟା;

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
    ଯଦି (!rdma->qp) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    ଯଦି (!rdma->tx_buf || !rdma->rx_buf) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ଯଦି (!rdma->tx_mr || !rdma->rx_mr) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ibv_gid local_gid;
    ଯଦି (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    rdma_local.qpn = rdma->qp->qp_ଅଙ୍କ;
    rdma_local.psn = rdma->qp->qp_ଅଙ୍କ & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    ଅଚଳ ଅକ୍ଷର * ver_str = "";
    ଯଦି (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } ଅନ୍ୟଥା ଯଦି (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
ତାର୍କିକ socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ଅଚଳ uint8_t * remote_gid) {
    // RESET -> INIT
    {
        ସଂରଚନା ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_ଅଙ୍କ        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        ଯଦି (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < RDMA_RX_DEPTH; i++) {
        ଯଦି (!rdma->post_rx(i)) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // INIT -> RTR
    {
        ସଂରଚନା ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_ଅଙ୍କ        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_ଅଙ୍କ   = rdma_local.ib_port;
        ଯଦି (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    // RTR -> RTS
    {
        ସଂରଚନା ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        ଯଦି (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Sleep until the completion channel has an event or the TCP peer closes.
ତାର୍କିକ socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    ସଂରଚନା pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    ଯଦି (poll(pfds, 2, -1) < 0) {
        ଫେରନ୍ତୁ errno == EINTR;
    }
    ଯଦି (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (pfds[0].revents & POLLIN) {
        ସଂରଚନା ibv_cq * ev_cq  = nullptr;
        ଶୂନ୍ୟ          * ev_ctx = nullptr;
        ଯଦି (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ socket_t::impl::rdma_poll(ସଂରଚନା ibv_cq * cq, ସଂରଚନା ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    ତାର୍କିକ armed = ମିଥ୍ୟା;
    ପାଇଁ (uint64_t s = 1; ; s++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = ibv_poll_cq(cq, 1, wc);
        ଯଦି (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            ଯଦି (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            ଫେରନ୍ତୁ wc->status == IBV_WC_SUCCESS;
        }
        ଯଦି (n < 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        ଯଦି (armed) {
            // armed and still empty: sleep until the next completion
            ଯଦି (!rdma_wait_event()) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            armed = ମିଥ୍ୟା;
            ଜାରି;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        ଯଦି (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            ଯଦି (ibv_req_notify_cq(cq, 0) != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            armed = ସତ୍ୟ;
            ଜାରି;
        }
        ଯଦି ((s & 0xFFFFF) == 0) {
            ଯଦି (tcp_peer_closed()) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
    }
}

ତାର୍କିକ socket_t::impl::rdma_send(ଅଚଳ ଶୂନ୍ୟ * data, size_t size) {
    rdma_conn * c = rdma.get();
    ଅଚଳ uint8_t * src = (ଅଚଳ uint8_t *)data;
    size_t rem = size;
    ଯେତେବେଳେ (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        ସଂରଚନା ibv_sge sge = {};
        ସଂରଚନା ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.ଅଙ୍କ_sge = 1;

        ଯଦି (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } ଅନ୍ୟଥା {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        ଯଦି (ibv_post_send(c->qp, &wr, &bad) != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
        c->last_active = std::chrono::steady_clock::now();
        ସଂରଚନା ibv_wc wc;
        ଯଦି (!rdma_poll(c->scq, &wc)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        src += chunk;
        rem -= chunk;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ socket_t::impl::rdma_recv(ଶୂନ୍ୟ * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    ଯେତେବେଳେ (rem > 0) {
        ସଂରଚନା ibv_wc wc;
        ଯଦି (!rdma_poll(c->rcq, &wc)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        ଯଦି (!c->post_rx(slot)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

        dst += got;
        rem -= got;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

ତାର୍କିକ socket_t::impl::send_data(ଅଚଳ ଶୂନ୍ୟ * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    ଯଦି (use_rdma) {
        ଫେରନ୍ତୁ rdma->send(data, size);
    }
#ନଚେତଯଦି defined(GGML_RPC_RDMA)
    ଯଦି (use_rdma) {
        ଫେରନ୍ତୁ rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    ଯେତେବେଳେ (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (ଅଚଳ ଅକ୍ଷର *)data + bytes_sent, size_to_send, 0);
        ଯଦି (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        bytes_sent += (size_t)n;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ତାର୍କିକ socket_t::impl::recv_data(ଶୂନ୍ୟ * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    ଯଦି (use_rdma) {
        ଫେରନ୍ତୁ rdma->recv(data, size);
    }
#ନଚେତଯଦି defined(GGML_RPC_RDMA)
    ଯଦି (use_rdma) {
        ଫେରନ୍ତୁ rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    ଯେତେବେଳେ (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (ଅକ୍ଷର *)data + bytes_recv, size_to_recv, 0);
        ଯଦି (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        bytes_recv += (size_t)n;
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ଶୂନ୍ୟ socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    ଯଦି (std::getenv("GGML_RPC_NO_RDMA")) {
        ଫେରନ୍ତୁ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    ଯଦି (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    ଯଦି (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  ଅନ୍ୟଥା
    rdma_local = {};
    ଯଦି (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, ଆକାର(rc));
    } ଅନ୍ୟଥା {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

ଶୂନ୍ୟ socket_t::impl::update_caps(ଅଚଳ uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    ତାର୍କିକ remote_rdma = ମିଥ୍ୟା;
    ପାଇଁ (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    ଯଦି (!rdma || !remote_rdma) {
        ଯଦି (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        ଫେରନ୍ତୁ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    ତାର୍କିକ activated = rdma->activate(remote_caps);
#  ଅନ୍ୟଥା
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, ଆକାର(rc));
    ତାର୍କିକ activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    ଯଦି (activated) {
        use_rdma = ସତ୍ୟ;
    } ଅନ୍ୟଥା {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#ଅନ୍ୟଥା
    (ଶୂନ୍ୟ)remote_caps;
#endif // GGML_RPC_RDMA
}

ତାର୍କିକ socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    ଯଦି (use_rdma) {
        ଫେରନ୍ତୁ rdma->flush();
    }
#endif
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;

ତାର୍କିକ socket_t::send_data(ଅଚଳ ଶୂନ୍ୟ * data, size_t size) {
    ଫେରନ୍ତୁ pimpl->send_data(data, size);
}

ତାର୍କିକ socket_t::recv_data(ଶୂନ୍ୟ * data, size_t size) {
    ଫେରନ୍ତୁ pimpl->recv_data(data, size);
}

ତାର୍କିକ socket_t::flush() {
    ଫେରନ୍ତୁ pimpl->flush();
}

ଶୂନ୍ୟ socket_t::get_caps(uint8_t * local_caps) {
    ଫେରନ୍ତୁ pimpl->get_caps(local_caps);
}

ଶୂନ୍ୟ socket_t::update_caps(ଅଚଳ uint8_t * remote_caps) {
    ଫେରନ୍ତୁ pimpl->update_caps(remote_caps);
}

ଅଚଳ ଅକ୍ଷର * socket_t::transport_name() ଅଚଳ {
    ଫେରନ୍ତୁ pimpl->use_rdma ? "RDMA" : "TCP";
}

ସ୍ଥିର ତାର୍କିକ is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    ଫେରନ୍ତୁ sockfd != INVALID_SOCKET;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ sockfd >= 0;
#endif
}

ସ୍ଥିର ତାର୍କିକ set_no_delay(sockfd_t sockfd) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (ଅକ୍ଷର *)&flag, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));
    ଫେରନ୍ତୁ ret == 0;
}

ସ୍ଥିର ତାର୍କିକ set_reuse_addr(sockfd_t sockfd) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା flag = 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (ଅକ୍ଷର *)&flag, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));
    ଫେରନ୍ତୁ ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, ଖାଲି, ଖାଲି);
    ଯଦି (!is_valid_fd(client_socket_fd)) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ଫେରନ୍ତୁ nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    ଫେରନ୍ତୁ socket_ptr(ନୂତନ socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ଅଚଳ ଅକ୍ଷର * host, ପୂର୍ଣ୍ଣସଂଖ୍ୟା port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    ଯଦି (!is_valid_fd(sockfd)) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        ଫେରନ୍ତୁ nullptr;
    }
    ସଂରଚନା sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    ଯଦି (bind(sockfd, (ସଂରଚନା sockaddr *) &serv_addr, ଆକାର(serv_addr)) < 0) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (listen(sockfd, 1) < 0) {
        ଫେରନ୍ତୁ nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    ଫେରନ୍ତୁ socket_ptr(ନୂତନ socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ଅଚଳ ଅକ୍ଷର * host, ପୂର୍ଣ୍ଣସଂଖ୍ୟା port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    ଯଦି (!is_valid_fd(sockfd)) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଯଦି (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ଫେରନ୍ତୁ nullptr;
    }
    ସଂରଚନା sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ସଂରଚନା hostent * server = gethostbyname(host);
    ଯଦି (server == ଖାଲି) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        ଫେରନ୍ତୁ nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    ଯଦି (::connect(sockfd, (ସଂରଚନା sockaddr *)&addr, ଆକାର(addr)) < 0) {
        ଫେରନ୍ତୁ nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    ଫେରନ୍ତୁ socket_ptr(ନୂତନ socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
ସ୍ଥିର std::mutex g_rpc_transport_mu;
ସ୍ଥିର ତାର୍କିକ g_rpc_transport_wsa_started = ମିଥ୍ୟା;
#endif

ତାର୍କିକ rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    ଯଦି (g_rpc_transport_wsa_started) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    WSADATA wsaData;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    ଯଦି (res != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    g_rpc_transport_wsa_started = ସତ୍ୟ;
    ଫେରନ୍ତୁ ସତ୍ୟ;
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ ସତ୍ୟ;
#endif
}

ଶୂନ୍ୟ rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    ଯଦି (!g_rpc_transport_wsa_started) {
        ଫେରନ୍ତୁ;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = ମିଥ୍ୟା;
#endif
}

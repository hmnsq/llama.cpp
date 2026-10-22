#ಅನ್ತರ್ಭಾವ "transport.h"
#ಅನ್ತರ್ಭಾವ "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#ಅನ್ಯಥಾ
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <optional>

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
ಪ್ರಕಾರನಿರ್ದೇಶ SOCKET sockfd_t;
using ssize_t = __int64;
#ಅನ್ಯಥಾ
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
ಸ್ಥಿರ constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#ಯದಿ defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ಸ್ಥಿರ constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ಸ್ಥಿರ constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

ಸಂರಚನಾ rdma_conn {
    ಸಂರಚನಾ ibv_context * ctx = nullptr;
    ಸಂರಚನಾ ibv_pd * pd  = nullptr;
    ಸಂರಚನಾ ibv_cq * scq = nullptr;   // send completions
    ಸಂರಚನಾ ibv_cq * rcq = nullptr;   // recv completions
    ಸಂರಚನಾ ibv_qp * qp  = nullptr;
    ಸಂರಚನಾ ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    ಶೂನ್ಯ          * tx_buf = nullptr;
    ಸಂರಚನಾ ibv_mr * tx_mr  = nullptr;

    ಶೂನ್ಯ          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    ಸಂರಚನಾ ibv_mr * rx_mr  = nullptr;
    ಪೂರ್ಣಾಙ್ಕ             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(ಪೂರ್ಣಾಙ್ಕ i) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    ತಾರ್ಕಿಕ post_rx(ಪೂರ್ಣಾಙ್ಕ i) {
        ಸಂರಚನಾ ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        ಸಂರಚನಾ ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ಅಂಕೆ_sge = 1;
        ಪ್ರತ್ಯಾಗಮ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        ಯದಿ (tx_mr) ibv_dereg_mr(tx_mr);
        ಯದಿ (rx_mr) ibv_dereg_mr(rx_mr);
        ಮುಕ್ತ(tx_buf);
        ಮುಕ್ತ(rx_buf);
        ಯದಿ (qp)  ibv_destroy_qp(qp);
        ಯದಿ (scq) ibv_destroy_cq(scq);
        ಯದಿ (rcq) ibv_destroy_cq(rcq);
        ಯದಿ (ch)  ibv_destroy_comp_channel(ch);
        ಯದಿ (pd)  ibv_dealloc_pd(pd);
        ಯದಿ (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
ಸಂರಚನಾ rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    ಪೂರ್ಣಾಙ್ಕ      gid_idx = 0;
    ಗಣನಾ ibv_mtu path_mtu = IBV_MTU_1024;
};

ಸಂರಚನಾ rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(ಪರಿಮಾಣ(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

ಸಂರಚನಾ socket_t::impl {
    impl(sockfd_t fd) : use_rdma(ಅಸತ್ಯ), fd(fd) {}
    ~impl();
    ತಾರ್ಕಿಕ send_data(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size);
    ತಾರ್ಕಿಕ recv_data(ಶೂನ್ಯ * data, size_t size);
    ತಾರ್ಕಿಕ flush();
    ಶೂನ್ಯ get_caps(uint8_t * local_caps);
    ಶೂನ್ಯ update_caps(ಧ್ರುವ uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  ಅನ್ಯಥಾ
    ತಾರ್ಕಿಕ rdma_probe();
    ತಾರ್ಕಿಕ rdma_send(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size);
    ತಾರ್ಕಿಕ rdma_recv(ಶೂನ್ಯ * data, size_t size);
    ತಾರ್ಕಿಕ tcp_peer_closed();
    ತಾರ್ಕಿಕ rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ಧ್ರುವ uint8_t * remote_gid);
    ತಾರ್ಕಿಕ rdma_poll(ಸಂರಚನಾ ibv_cq * cq, ಸಂರಚನಾ ibv_wc * wc);
    ತಾರ್ಕಿಕ rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    ತಾರ್ಕಿಕ     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, ಏತತ್->fd);
#ifdef _WIN32
    ಯದಿ (fd != INVALID_SOCKET) closesocket(ಏತತ್->fd);
#ಅನ್ಯಥಾ
    ಯದಿ (fd >= 0) close(ಏತತ್->fd);
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
    socklen_t addr_len = ಪರಿಮಾಣ(addr);
    ಯದಿ (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        ಪ್ರತ್ಯಾಗಮ std::nullopt;
    }
    rdma_gid_t target = {};
    ಯದಿ (addr.ss_family == AF_INET) {
        ಧ್ರುವ auto * a = reinterpret_cast<ಧ್ರುವ sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        ಪ್ರತ್ಯಾಗಮ target;
    }
    ಯದಿ (addr.ss_family == AF_INET6) {
        ಧ್ರುವ auto * a = reinterpret_cast<ಧ್ರುವ sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        ಪ್ರತ್ಯಾಗಮ target;
    }
    ಪ್ರತ್ಯಾಗಮ std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

ತಾರ್ಕಿಕ socket_t::impl::tcp_peer_closed() {
    ಯದಿ (fd < 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
#ifndef _WIN32
    ಸಂರಚನಾ pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    ಪೂರ್ಣಾಙ್ಕ r = poll(&pfd, 1, 0);
    ಪ್ರತ್ಯಾಗಮ r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
#endif
}

ತಾರ್ಕಿಕ socket_t::impl::rdma_probe() {
    ಧ್ರುವ ವರ್ಣ * dev_env = std::getenv("GGML_RDMA_DEV");
    ಧ್ರುವ ವರ್ಣ * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    ಯದಿ (!target_gid) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ uint8_t ib_port = 1;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&ಅಂಕೆ_devs);
    ಯದಿ (!devs || ಅಂಕೆ_devs == 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    ibv_context * ibctx = nullptr;
    ಧ್ರುವ ವರ್ಣ * matched_dev = nullptr;
    ಪೂರ್ಣಾಙ್ಕ gid_idx = gid_env ? atoi(gid_env) : -1;
    ಪೂರ್ಣಾಙ್ಕ gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ d = 0; d < ಅಂಕೆ_devs; d++) {
        ಧ್ರುವ ವರ್ಣ * dn = ibv_get_device_name(devs[d]);
        ಯದಿ (dev_env && strcmp(dev_env, dn) != 0) ಅನುವರ್ತ;

        ibv_context * ctx = ibv_open_device(devs[d]);
        ಯದಿ (!ctx) ಅನುವರ್ತ;

        ibv_port_attr pa;
        ಯದಿ (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); ಅನುವರ್ತ; }

        ಪೂರ್ಣಾಙ್ಕ found_gid = gid_idx;
        ಪೂರ್ಣಾಙ್ಕ found_version = IBV_GID_TYPE_IB;
        ಯದಿ (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            ಪೂರ್ಣಾಙ್ಕ v2_idx = -1;
            ಪೂರ್ಣಾಙ್ಕ v1_idx = -1;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                ಯದಿ (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) ಅನುವರ್ತ;
                ಯದಿ (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) ಅನುವರ್ತ;
                ಯದಿ (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } ಅನ್ಯಥಾ ಯದಿ (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            ಯದಿ (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } ಅನ್ಯಥಾ ಯದಿ (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } ಅನ್ಯಥಾ {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            ಯದಿ (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        ಯದಿ (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            ಭಙ್ಗ;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    ಯದಿ (!ibctx) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    ಯದಿ (!rdma->pd) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    ಯದಿ (!rdma->scq || !rdma->rcq) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

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
    ಯದಿ (!rdma->qp) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    ಯದಿ (!rdma->tx_buf || !rdma->rx_buf) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ಯದಿ (!rdma->tx_mr || !rdma->rx_mr) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    ibv_gid local_gid;
    ಯದಿ (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

    rdma_local.qpn = rdma->qp->qp_ಅಂಕೆ;
    rdma_local.psn = rdma->qp->qp_ಅಂಕೆ & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    ಧ್ರುವ ವರ್ಣ * ver_str = "";
    ಯದಿ (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } ಅನ್ಯಥಾ ಯದಿ (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
ತಾರ್ಕಿಕ socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ಧ್ರುವ uint8_t * remote_gid) {
    // RESET -> INIT
    {
        ಸಂರಚನಾ ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_ಅಂಕೆ        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        ಯದಿ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < RDMA_RX_DEPTH; i++) {
        ಯದಿ (!rdma->post_rx(i)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // INIT -> RTR
    {
        ಸಂರಚನಾ ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_ಅಂಕೆ        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_ಅಂಕೆ   = rdma_local.ib_port;
        ಯದಿ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    // RTR -> RTS
    {
        ಸಂರಚನಾ ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        ಯದಿ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// Sleep until the completion channel has an event or the TCP peer closes.
ತಾರ್ಕಿಕ socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    ಸಂರಚನಾ pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    ಯದಿ (poll(pfds, 2, -1) < 0) {
        ಪ್ರತ್ಯಾಗಮ errno == EINTR;
    }
    ಯದಿ (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (pfds[0].revents & POLLIN) {
        ಸಂರಚನಾ ibv_cq * ev_cq  = nullptr;
        ಶೂನ್ಯ          * ev_ctx = nullptr;
        ಯದಿ (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ socket_t::impl::rdma_poll(ಸಂರಚನಾ ibv_cq * cq, ಸಂರಚನಾ ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    ತಾರ್ಕಿಕ armed = ಅಸತ್ಯ;
    ಪ್ರತಿ (uint64_t s = 1; ; s++) {
        ಪೂರ್ಣಾಙ್ಕ n = ibv_poll_cq(cq, 1, wc);
        ಯದಿ (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            ಯದಿ (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            ಪ್ರತ್ಯಾಗಮ wc->status == IBV_WC_SUCCESS;
        }
        ಯದಿ (n < 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        ಯದಿ (armed) {
            // armed and still empty: sleep until the next completion
            ಯದಿ (!rdma_wait_event()) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            armed = ಅಸತ್ಯ;
            ಅನುವರ್ತ;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        ಯದಿ (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            ಯದಿ (ibv_req_notify_cq(cq, 0) != 0) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            armed = ಸತ್ಯ;
            ಅನುವರ್ತ;
        }
        ಯದಿ ((s & 0xFFFFF) == 0) {
            ಯದಿ (tcp_peer_closed()) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    }
}

ತಾರ್ಕಿಕ socket_t::impl::rdma_send(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size) {
    rdma_conn * c = rdma.get();
    ಧ್ರುವ uint8_t * src = (ಧ್ರುವ uint8_t *)data;
    size_t rem = size;
    ಯಾವತ್ (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        ಸಂರಚನಾ ibv_sge sge = {};
        ಸಂರಚನಾ ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.ಅಂಕೆ_sge = 1;

        ಯದಿ (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } ಅನ್ಯಥಾ {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        ಯದಿ (ibv_post_send(c->qp, &wr, &bad) != 0) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        c->last_active = std::chrono::steady_clock::now();
        ಸಂರಚನಾ ibv_wc wc;
        ಯದಿ (!rdma_poll(c->scq, &wc)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        src += chunk;
        rem -= chunk;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ socket_t::impl::rdma_recv(ಶೂನ್ಯ * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    ಯಾವತ್ (rem > 0) {
        ಸಂರಚನಾ ibv_wc wc;
        ಯದಿ (!rdma_poll(c->rcq, &wc)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        ಪೂರ್ಣಾಙ್ಕ slot = (ಪೂರ್ಣಾಙ್ಕ)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        ಯದಿ (!c->post_rx(slot)) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;

        dst += got;
        rem -= got;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

ತಾರ್ಕಿಕ socket_t::impl::send_data(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    ಯದಿ (use_rdma) {
        ಪ್ರತ್ಯಾಗಮ rdma->send(data, size);
    }
#ನೋಚೇದ್ಯದಿ defined(GGML_RPC_RDMA)
    ಯದಿ (use_rdma) {
        ಪ್ರತ್ಯಾಗಮ rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    ಯಾವತ್ (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (ಧ್ರುವ ವರ್ಣ *)data + bytes_sent, size_to_send, 0);
        ಯದಿ (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        bytes_sent += (size_t)n;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ತಾರ್ಕಿಕ socket_t::impl::recv_data(ಶೂನ್ಯ * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    ಯದಿ (use_rdma) {
        ಪ್ರತ್ಯಾಗಮ rdma->recv(data, size);
    }
#ನೋಚೇದ್ಯದಿ defined(GGML_RPC_RDMA)
    ಯದಿ (use_rdma) {
        ಪ್ರತ್ಯಾಗಮ rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    ಯಾವತ್ (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (ವರ್ಣ *)data + bytes_recv, size_to_recv, 0);
        ಯದಿ (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        bytes_recv += (size_t)n;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಶೂನ್ಯ socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    ಯದಿ (std::getenv("GGML_RPC_NO_RDMA")) {
        ಪ್ರತ್ಯಾಗಮ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    ಯದಿ (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    ಯದಿ (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  ಅನ್ಯಥಾ
    rdma_local = {};
    ಯದಿ (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, ಪರಿಮಾಣ(rc));
    } ಅನ್ಯಥಾ {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

ಶೂನ್ಯ socket_t::impl::update_caps(ಧ್ರುವ uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    ತಾರ್ಕಿಕ remote_rdma = ಅಸತ್ಯ;
    ಪ್ರತಿ (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    ಯದಿ (!rdma || !remote_rdma) {
        ಯದಿ (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        ಪ್ರತ್ಯಾಗಮ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    ತಾರ್ಕಿಕ activated = rdma->activate(remote_caps);
#  ಅನ್ಯಥಾ
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, ಪರಿಮಾಣ(rc));
    ತಾರ್ಕಿಕ activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    ಯದಿ (activated) {
        use_rdma = ಸತ್ಯ;
    } ಅನ್ಯಥಾ {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#ಅನ್ಯಥಾ
    (ಶೂನ್ಯ)remote_caps;
#endif // GGML_RPC_RDMA
}

ತಾರ್ಕಿಕ socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    ಯದಿ (use_rdma) {
        ಪ್ರತ್ಯಾಗಮ rdma->flush();
    }
#endif
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = ಸ್ವಭಾವ;

ತಾರ್ಕಿಕ socket_t::send_data(ಧ್ರುವ ಶೂನ್ಯ * data, size_t size) {
    ಪ್ರತ್ಯಾಗಮ pimpl->send_data(data, size);
}

ತಾರ್ಕಿಕ socket_t::recv_data(ಶೂನ್ಯ * data, size_t size) {
    ಪ್ರತ್ಯಾಗಮ pimpl->recv_data(data, size);
}

ತಾರ್ಕಿಕ socket_t::flush() {
    ಪ್ರತ್ಯಾಗಮ pimpl->flush();
}

ಶೂನ್ಯ socket_t::get_caps(uint8_t * local_caps) {
    ಪ್ರತ್ಯಾಗಮ pimpl->get_caps(local_caps);
}

ಶೂನ್ಯ socket_t::update_caps(ಧ್ರುವ uint8_t * remote_caps) {
    ಪ್ರತ್ಯಾಗಮ pimpl->update_caps(remote_caps);
}

ಧ್ರುವ ವರ್ಣ * socket_t::transport_name() ಧ್ರುವ {
    ಪ್ರತ್ಯಾಗಮ pimpl->use_rdma ? "RDMA" : "TCP";
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    ಪ್ರತ್ಯಾಗಮ sockfd != INVALID_SOCKET;
#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ sockfd >= 0;
#endif
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ set_no_delay(sockfd_t sockfd) {
    ಪೂರ್ಣಾಙ್ಕ flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    ಪೂರ್ಣಾಙ್ಕ ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (ವರ್ಣ *)&flag, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪ್ರತ್ಯಾಗಮ ret == 0;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ set_reuse_addr(sockfd_t sockfd) {
    ಪೂರ್ಣಾಙ್ಕ flag = 1;
    ಪೂರ್ಣಾಙ್ಕ ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (ವರ್ಣ *)&flag, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪ್ರತ್ಯಾಗಮ ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, ರಿಕ್ತ, ರಿಕ್ತ);
    ಯದಿ (!is_valid_fd(client_socket_fd)) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    ಪ್ರತ್ಯಾಗಮ socket_ptr(ನೂತನ socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ಧ್ರುವ ವರ್ಣ * host, ಪೂರ್ಣಾಙ್ಕ port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    ಯದಿ (!is_valid_fd(sockfd)) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಸಂರಚನಾ sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    ಯದಿ (bind(sockfd, (ಸಂರಚನಾ sockaddr *) &serv_addr, ಪರಿಮಾಣ(serv_addr)) < 0) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (listen(sockfd, 1) < 0) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    ಪ್ರತ್ಯಾಗಮ socket_ptr(ನೂತನ socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ಧ್ರುವ ವರ್ಣ * host, ಪೂರ್ಣಾಙ್ಕ port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    ಯದಿ (!is_valid_fd(sockfd)) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಯದಿ (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಸಂರಚನಾ sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ಸಂರಚನಾ hostent * server = gethostbyname(host);
    ಯದಿ (server == ರಿಕ್ತ) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    ಯದಿ (::connect(sockfd, (ಸಂರಚನಾ sockaddr *)&addr, ಪರಿಮಾಣ(addr)) < 0) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    ಪ್ರತ್ಯಾಗಮ socket_ptr(ನೂತನ socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
ಸ್ಥಿರ std::mutex g_rpc_transport_mu;
ಸ್ಥಿರ ತಾರ್ಕಿಕ g_rpc_transport_wsa_started = ಅಸತ್ಯ;
#endif

ತಾರ್ಕಿಕ rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    ಯದಿ (g_rpc_transport_wsa_started) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    WSADATA wsaData;
    ಪೂರ್ಣಾಙ್ಕ res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    ಯದಿ (res != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    g_rpc_transport_wsa_started = ಸತ್ಯ;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
#endif
}

ಶೂನ್ಯ rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    ಯದಿ (!g_rpc_transport_wsa_started) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = ಅಸತ್ಯ;
#endif
}

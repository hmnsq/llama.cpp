#ಅನ್ತರ್ಭಾವ "transport.h"
#ಅನ್ತರ್ಭಾವ "log.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#  define WIN32_LEAN_AND_ಸರಾಸರಿ
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#ಇಲ್ಲದಿದ್ದರೆ
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <optional>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA
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
#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ SOCKET sockfd_t;
ಬಳಸಿ ssize_t = __int64;
#ಇಲ್ಲದಿದ್ದರೆ
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ sockfd_t;
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA
ಸ್ಥಿರ constexpr size_t RDMA_GID_ಗಾತ್ರ = 16;            // RoCE GID / IB GID is always 16 bytes
ಬಳಸಿ rdma_gid_t = std::array<uint8_t, RDMA_GID_ಗಾತ್ರ>;
#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA

#ಆದರೆ defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ಸ್ಥಿರ constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ    RDMA_RX_ಆಳ = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ಸ್ಥಿರ constexpr ಸ್ವಯಂಚಾಲಿತ   RDMA_SPIN_ಸಮಯ = std::chrono::milliseconds(100);

ರಚನೆ rdma_conn {
    ರಚನೆ ibv_context * ctx = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ibv_pd * pd  = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ibv_cq * scq = ಶೂನ್ಯಸೂಚಕ;   // send completions
    ರಚನೆ ibv_cq * rcq = ಶೂನ್ಯಸೂಚಕ;   // recv completions
    ರಚನೆ ibv_qp * qp  = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ibv_comp_channel * ch = ಶೂನ್ಯಸೂಚಕ; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point ಕೊನೆಯ_ಸಕ್ರಿಯ; // last completion or posted send

    ಶೂನ್ಯ          * tx_buf = ಶೂನ್ಯಸೂಚಕ;
    ರಚನೆ ibv_mr * tx_mr  = ಶೂನ್ಯಸೂಚಕ;

    ಶೂನ್ಯ          * rx_buf = ಶೂನ್ಯಸೂಚಕ; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    ರಚನೆ ibv_mr * rx_mr  = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ             rx_ತಲೆ = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_ಸ್ಥಾನ(ಪೂರ್ಣಾಂಕ i) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    ತಾರ್ಕಿಕ post_rx(ಪೂರ್ಣಾಂಕ i) {
        ರಚನೆ ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_ಸ್ಥಾನ(i);
        sge.ಉದ್ದಳತೆ = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        ರಚನೆ ibv_recv_wr wr = {}, * bad = ಶೂನ್ಯಸೂಚಕ;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ಅಂಕೆ_sge = 1;
        ಹಿಂತಿರುಗಿ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        ಆದರೆ (tx_mr) ibv_dereg_mr(tx_mr);
        ಆದರೆ (rx_mr) ibv_dereg_mr(rx_mr);
        ಬಿಡುಗಡೆ(tx_buf);
        ಬಿಡುಗಡೆ(rx_buf);
        ಆದರೆ (qp)  ibv_destroy_qp(qp);
        ಆದರೆ (scq) ibv_destroy_cq(scq);
        ಆದರೆ (rcq) ibv_destroy_cq(rcq);
        ಆದರೆ (ch)  ibv_destroy_comp_channel(ch);
        ಆದರೆ (pd)  ibv_dealloc_pd(pd);
        ಆದರೆ (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
ರಚನೆ rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_ಗಾತ್ರ] = {};
    uint8_t  ib_ಪೋರ್ಟ್ = 0;
    ಪೂರ್ಣಾಂಕ      gid_idx = 0;
    ಎಣಿಕೆ ibv_mtu ಮಾರ್ಗ_mtu = IBV_MTU_1024;
};

ರಚನೆ rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_ಗಾತ್ರ];
};

static_assert(ಗಾತ್ರ(rdma_caps) == RPC_CONN_CAPS_ಗಾತ್ರ, "rdma_caps must match conn_caps size");

#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

ರಚನೆ socket_t::impl {
    impl(sockfd_t fd) : use_rdma(ಸುಳ್ಳು), fd(fd) {}
    ~impl();
    ತಾರ್ಕಿಕ send_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size);
    ತಾರ್ಕಿಕ recv_ದತ್ತಾಂಶ(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size);
    ತಾರ್ಕಿಕ flush();
    ಶೂನ್ಯ get_caps(uint8_t * local_caps);
    ಶೂನ್ಯ update_caps(ಸ್ಥಿರಾಂಕ uint8_t * remote_caps);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_ಗುರಿ_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  ಇಲ್ಲದಿದ್ದರೆ
    ತಾರ್ಕಿಕ rdma_probe();
    ತಾರ್ಕಿಕ rdma_send(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size);
    ತಾರ್ಕಿಕ rdma_recv(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size);
    ತಾರ್ಕಿಕ tcp_peer_closed();
    ತಾರ್ಕಿಕ rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ಸ್ಥಿರಾಂಕ uint8_t * remote_gid);
    ತಾರ್ಕಿಕ rdma_poll(ರಚನೆ ibv_cq * cq, ರಚನೆ ibv_wc * wc);
    ತಾರ್ಕಿಕ rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA
    ತಾರ್ಕಿಕ     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA
    rdma.reset();
#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, ಇದು->fd);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಆದರೆ (fd != INVALID_SOCKET) closesocket(ಇದು->fd);
#ಇಲ್ಲದಿದ್ದರೆ
    ಆದರೆ (fd >= 0) close(ಇದು->fd);
#ಆದರೆ_ಅಂತ್ಯ
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_ಗುರಿ_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = ಗಾತ್ರ(addr);
    ಆದರೆ (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        ಹಿಂತಿರುಗಿ std::nullopt;
    }
    rdma_gid_t ಗುರಿ = {};
    ಆದರೆ (addr.ss_family == AF_INET) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * a = reinterpret_cast<ಸ್ಥಿರಾಂಕ sockaddr_ಒಳ *>(&addr);
        ಗುರಿ[10] = 0xff;
        ಗುರಿ[11] = 0xff;
        ಸ್ಮರಣೆ_ನಕಲು(&ಗುರಿ[12], &a->sin_addr, 4);
        ಹಿಂತಿರುಗಿ ಗುರಿ;
    }
    ಆದರೆ (addr.ss_family == AF_INET6) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * a = reinterpret_cast<ಸ್ಥಿರಾಂಕ sockaddr_in6 *>(&addr);
        ಸ್ಮರಣೆ_ನಕಲು(ಗುರಿ.ದತ್ತಾಂಶ(), &a->sin6_addr, RDMA_GID_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ಗುರಿ;
    }
    ಹಿಂತಿರುಗಿ std::nullopt;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_RPC_RDMA_APPLE

ತಾರ್ಕಿಕ socket_t::impl::tcp_peer_closed() {
    ಆದರೆ (fd < 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ _WIN32
    ರಚನೆ pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    ಪೂರ್ಣಾಂಕ r = poll(&pfd, 1, 0);
    ಹಿಂತಿರುಗಿ r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
}

ತಾರ್ಕಿಕ socket_t::impl::rdma_probe() {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dev_env = std::getenv("GGML_RDMA_DEV");
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * gid_env = std::getenv("GGML_RDMA_GID");

    ಸ್ವಯಂಚಾಲಿತ ಗುರಿ_gid = rdma_build_ಗುರಿ_gid();
    ಆದರೆ (!ಗುರಿ_gid) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint8_t ib_ಪೋರ್ಟ್ = 1;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&ಅಂಕೆ_devs);
    ಆದರೆ (!devs || ಅಂಕೆ_devs == 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ibv_context * ibctx = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * matched_dev = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ gid_idx = gid_env ? atoi(gid_env) : -1;
    ಪೂರ್ಣಾಂಕ gid_ಆವೃತ್ತಿ = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < ಅಂಕೆ_devs; d++) {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dn = ibv_get_device_name(devs[d]);
        ಆದರೆ (dev_env && strcmp(dev_env, dn) != 0) ಮುಂದುವರಿ;

        ibv_context * ctx = ibv_open_device(devs[d]);
        ಆದರೆ (!ctx) ಮುಂದುವರಿ;

        ibv_ಪೋರ್ಟ್_attr pa;
        ಆದರೆ (ibv_query_ಪೋರ್ಟ್(ctx, ib_ಪೋರ್ಟ್, &pa) != 0) { ibv_close_device(ctx); ಮುಂದುವರಿ; }

        ಪೂರ್ಣಾಂಕ ಸಿಕ್ಕಿತು_gid = gid_idx;
        ಪೂರ್ಣಾಂಕ ಸಿಕ್ಕಿತು_ಆವೃತ್ತಿ = IBV_GID_TYPE_IB;
        ಆದರೆ (ಸಿಕ್ಕಿತು_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            ಪೂರ್ಣಾಂಕ v2_idx = -1;
            ಪೂರ್ಣಾಂಕ v1_idx = -1;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_ನಮೂದು ನಮೂದು = {};
                ಆದರೆ (ibv_query_gid_ex(ctx, ib_ಪೋರ್ಟ್, i, &ನಮೂದು, 0) != 0) ಮುಂದುವರಿ;
                ಆದರೆ (memcmp(ನಮೂದು.gid.raw, ಗುರಿ_gid->ದತ್ತಾಂಶ(), RDMA_GID_ಗಾತ್ರ) != 0) ಮುಂದುವರಿ;
                ಆದರೆ (ನಮೂದು.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ನಮೂದು.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            ಆದರೆ (v2_idx >= 0) {
                ಸಿಕ್ಕಿತು_gid = v2_idx;
                ಸಿಕ್ಕಿತು_ಆವೃತ್ತಿ = IBV_GID_TYPE_ROCE_V2;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (v1_idx >= 0) {
                ಸಿಕ್ಕಿತು_gid = v1_idx;
                ಸಿಕ್ಕಿತು_ಆವೃತ್ತಿ = IBV_GID_TYPE_ROCE_V1;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_ನಮೂದು ನಮೂದು = {};
            ಆದರೆ (ibv_query_gid_ex(ctx, ib_ಪೋರ್ಟ್, ಸಿಕ್ಕಿತು_gid, &ನಮೂದು, 0) == 0) {
                ಸಿಕ್ಕಿತು_ಆವೃತ್ತಿ = ನಮೂದು.gid_type;
            }
        }
        ಆದರೆ (ಸಿಕ್ಕಿತು_gid >= 0) {
            ibctx = ctx;
            gid_idx = ಸಿಕ್ಕಿತು_gid;
            gid_ಆವೃತ್ತಿ = ಸಿಕ್ಕಿತು_ಆವೃತ್ತಿ;
            matched_dev = dn;
            rdma_local.ಮಾರ್ಗ_mtu = pa.ಸಕ್ರಿಯ_mtu;
            ನಿಲ್ಲಿಸು;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    ಆದರೆ (!ibctx) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    rdma_local.ib_ಪೋರ್ಟ್ = ib_ಪೋರ್ಟ್;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    ಆದರೆ (!rdma->pd) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, ಶೂನ್ಯಸೂಚಕ, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_ಆಳ + 4, ಶೂನ್ಯಸೂಚಕ, rdma->ch, 0);
    ಆದರೆ (!rdma->scq || !rdma->rcq) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_ಆಳ + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_ದತ್ತಾಂಶ = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    ಆದರೆ (!rdma->qp) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    rdma->max_inline = qia.cap.max_inline_ದತ್ತಾಂಶ;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_ಆಳ) * RDMA_CHUNK);
    ಆದರೆ (!rdma->tx_buf || !rdma->rx_buf) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_ಆಳ) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ಆದರೆ (!rdma->tx_mr || !rdma->rx_mr) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ibv_gid local_gid;
    ಆದರೆ (ibv_query_gid(ibctx, ib_ಪೋರ್ಟ್, gid_idx, &local_gid) != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    rdma_local.qpn = rdma->qp->qp_ಅಂಕೆ;
    rdma_local.psn = rdma->qp->qp_ಅಂಕೆ & 0xffffff;
    ಸ್ಮರಣೆ_ನಕಲು(&rdma_local.gid, &local_gid, RDMA_GID_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ver_str = "";
    ಆದರೆ (gid_ಆವೃತ್ತಿ == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (gid_ಆವೃತ್ತಿ == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
ತಾರ್ಕಿಕ socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ಸ್ಥಿರಾಂಕ uint8_t * remote_gid) {
    // RESET -> INIT
    {
        ರಚನೆ ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.ಪೋರ್ಟ್_ಅಂಕೆ        = rdma_local.ib_ಪೋರ್ಟ್;
        a.pkey_ಸೂಚ್ಯಂಕ      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        ಆದರೆ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_ಸೂಚ್ಯಂಕ | IBV_QP_ಪೋರ್ಟ್ | IBV_QP_ACCESS_FLAGS) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < RDMA_RX_ಆಳ; i++) {
        ಆದರೆ (!rdma->post_rx(i)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // INIT -> RTR
    {
        ರಚನೆ ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.ಮಾರ್ಗ_mtu           = rdma_local.ಮಾರ್ಗ_mtu;
        a.dest_qp_ಅಂಕೆ        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        ಸ್ಮರಣೆ_ನಕಲು(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_ಗಾತ್ರ);
        a.ah_attr.grh.hop_ಎಲ್ಲೆ  = 1;
        a.ah_attr.grh.sgid_ಸೂಚ್ಯಂಕ = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.ಪೋರ್ಟ್_ಅಂಕೆ   = rdma_local.ib_ಪೋರ್ಟ್;
        ಆದರೆ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_ಮಾರ್ಗ_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    // RTR -> RTS
    {
        ರಚನೆ ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        ಆದರೆ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    rdma->ಕೊನೆಯ_ಸಕ್ರಿಯ = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.ಮಾರ್ಗ_mtu, RDMA_RX_ಆಳ);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Sleep until the completion channel has an event or the TCP peer closes.
ತಾರ್ಕಿಕ socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    ರಚನೆ pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    ಆದರೆ (poll(pfds, 2, -1) < 0) {
        ಹಿಂತಿರುಗಿ errno == EINTR;
    }
    ಆದರೆ (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (pfds[0].revents & POLLIN) {
        ರಚನೆ ibv_cq * ev_cq  = ಶೂನ್ಯಸೂಚಕ;
        ಶೂನ್ಯ          * ev_ctx = ಶೂನ್ಯಸೂಚಕ;
        ಆದರೆ (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ socket_t::impl::rdma_poll(ರಚನೆ ibv_cq * cq, ರಚನೆ ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    ತಾರ್ಕಿಕ armed = ಸುಳ್ಳು;
    ಗಾಗಿ (uint64_t s = 1; ; s++) {
        ಪೂರ್ಣಾಂಕ n = ibv_poll_cq(cq, 1, wc);
        ಆದರೆ (n > 0) {
            c->ಕೊನೆಯ_ಸಕ್ರಿಯ = std::chrono::steady_clock::now();
            ಆದರೆ (wc->ಸ್ಥಿತಿ_2 != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->ಸ್ಥಿತಿ_2, ibv_wc_ಸ್ಥಿತಿ_str(wc->ಸ್ಥಿತಿ_2), wc->vendor_err);
            }
            ಹಿಂತಿರುಗಿ wc->ಸ್ಥಿತಿ_2 == IBV_WC_SUCCESS;
        }
        ಆದರೆ (n < 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಆದರೆ (armed) {
            // armed and still empty: sleep until the next completion
            ಆದರೆ (!rdma_wait_event()) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            armed = ಸುಳ್ಳು;
            ಮುಂದುವರಿ;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        ಆದರೆ (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->ಕೊನೆಯ_ಸಕ್ರಿಯ > RDMA_SPIN_ಸಮಯ) {
            ಆದರೆ (ibv_req_notify_cq(cq, 0) != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            armed = ನಿಜ;
            ಮುಂದುವರಿ;
        }
        ಆದರೆ ((s & 0xFFFFF) == 0) {
            ಆದರೆ (tcp_peer_closed()) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }
}

ತಾರ್ಕಿಕ socket_t::impl::rdma_send(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    rdma_conn * c = rdma.get();
    ಸ್ಥಿರಾಂಕ uint8_t * src = (ಸ್ಥಿರಾಂಕ uint8_t *)ದತ್ತಾಂಶ;
    size_t rem = size;
    ಆಗ (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        ರಚನೆ ibv_sge sge = {};
        ರಚನೆ ibv_send_wr wr = {}, * bad = ಶೂನ್ಯಸೂಚಕ;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.ಅಂಕೆ_sge = 1;

        ಆದರೆ (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.ಉದ್ದಳತೆ = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಮರಣೆ_ನಕಲು(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.ಉದ್ದಳತೆ = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        ಆದರೆ (ibv_post_send(c->qp, &wr, &bad) != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        c->ಕೊನೆಯ_ಸಕ್ರಿಯ = std::chrono::steady_clock::now();
        ರಚನೆ ibv_wc wc;
        ಆದರೆ (!rdma_poll(c->scq, &wc)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        src += chunk;
        rem -= chunk;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ socket_t::impl::rdma_recv(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)ದತ್ತಾಂಶ;
    size_t rem = size;
    ಆಗ (rem > 0) {
        ರಚನೆ ibv_wc wc;
        ಆದರೆ (!rdma_poll(c->rcq, &wc)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ = (ಪೂರ್ಣಾಂಕ)wc.wr_id;
        size_t got = wc.byte_len;
        ಸ್ಮರಣೆ_ನಕಲು(dst, c->rx_ಸ್ಥಾನ(ಸ್ಥಾನ), got);

        ಆದರೆ (!c->post_rx(ಸ್ಥಾನ)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

        dst += got;
        rem -= got;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

#ಆದರೆ_ಅಂತ್ಯ // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA

ತಾರ್ಕಿಕ socket_t::impl::send_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_ಗಾತ್ರ(size).c_str());
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA_APPLE
    ಆದರೆ (use_rdma) {
        ಹಿಂತಿರುಗಿ rdma->send(ದತ್ತಾಂಶ, size);
    }
#ಅಥವಾಆದರೆ defined(GGML_RPC_RDMA)
    ಆದರೆ (use_rdma) {
        ಹಿಂತಿರುಗಿ rdma_send(ದತ್ತಾಂಶ, size);
    }
#ಆದರೆ_ಅಂತ್ಯ
    size_t ಬೈಟುಗಳು_ಕಳುಹಿಸಿದ = 0;
    ಆಗ (ಬೈಟುಗಳು_ಕಳುಹಿಸಿದ < size) {
        size_t ಗಾತ್ರ_ಗಮ್ಯ_send = std::min(size - ಬೈಟುಗಳು_ಕಳುಹಿಸಿದ, MAX_CHUNK_ಗಾತ್ರ);
        ssize_t n = send(fd, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)ದತ್ತಾಂಶ + ಬೈಟುಗಳು_ಕಳುಹಿಸಿದ, ಗಾತ್ರ_ಗಮ್ಯ_send, 0);
        ಆದರೆ (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           ಬೈಟುಗಳು_ಕಳುಹಿಸಿದ, ಗಾತ್ರ_ಗಮ್ಯ_send);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಬೈಟುಗಳು_ಕಳುಹಿಸಿದ += (size_t)n;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ socket_t::impl::recv_ದತ್ತಾಂಶ(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_ಗಾತ್ರ(size).c_str());
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA_APPLE
    ಆದರೆ (use_rdma) {
        ಹಿಂತಿರುಗಿ rdma->recv(ದತ್ತಾಂಶ, size);
    }
#ಅಥವಾಆದರೆ defined(GGML_RPC_RDMA)
    ಆದರೆ (use_rdma) {
        ಹಿಂತಿರುಗಿ rdma_recv(ದತ್ತಾಂಶ, size);
    }
#ಆದರೆ_ಅಂತ್ಯ
    size_t ಬೈಟುಗಳು_recv = 0;
    ಆಗ (ಬೈಟುಗಳು_recv < size) {
        size_t ಗಾತ್ರ_ಗಮ್ಯ_recv = std::min(size - ಬೈಟುಗಳು_recv, MAX_CHUNK_ಗಾತ್ರ);
        ssize_t n = recv(fd, (ಅಕ್ಷರ *)ದತ್ತಾಂಶ + ಬೈಟುಗಳು_recv, ಗಾತ್ರ_ಗಮ್ಯ_recv, 0);
        ಆದರೆ (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           ಬೈಟುಗಳು_recv, ಗಾತ್ರ_ಗಮ್ಯ_recv);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಬೈಟುಗಳು_recv += (size_t)n;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಶೂನ್ಯ socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_ಗಾತ್ರ);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA
    ಆದರೆ (std::getenv("GGML_RPC_NO_RDMA")) {
        ಹಿಂತಿರುಗಿ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    ಸ್ವಯಂಚಾಲಿತ ಗುರಿ_gid = rdma_build_ಗುರಿ_gid();
    ಆದರೆ (ಗುರಿ_gid) {
        rdma = apple_rdma::probe(fd, ಗುರಿ_gid->ದತ್ತಾಂಶ(), local_caps);
    }
    ಆದರೆ (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  ಇಲ್ಲದಿದ್ದರೆ
    rdma_local = {};
    ಆದರೆ (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        ಸ್ಮರಣೆ_ನಕಲು(rc.gid, rdma_local.gid, RDMA_GID_ಗಾತ್ರ);
        ಸ್ಮರಣೆ_ನಕಲು(local_caps, &rc, ಗಾತ್ರ(rc));
    } ಇಲ್ಲದಿದ್ದರೆ {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA
}

ಶೂನ್ಯ socket_t::impl::update_caps(ಸ್ಥಿರಾಂಕ uint8_t * remote_caps) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    ತಾರ್ಕಿಕ remote_rdma = ಸುಳ್ಳು;
    ಗಾಗಿ (size_t i = 0; i < RPC_CONN_CAPS_ಗಾತ್ರ; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    ಆದರೆ (!rdma || !remote_rdma) {
        ಆದರೆ (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        ಹಿಂತಿರುಗಿ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    ತಾರ್ಕಿಕ activated = rdma->activate(remote_caps);
#  ಇಲ್ಲದಿದ್ದರೆ
    rdma_caps rc = {};
    ಸ್ಮರಣೆ_ನಕಲು(&rc, remote_caps, ಗಾತ್ರ(rc));
    ತಾರ್ಕಿಕ activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    ಆದರೆ (activated) {
        use_rdma = ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#ಇಲ್ಲದಿದ್ದರೆ
    (ಶೂನ್ಯ)remote_caps;
#ಆದರೆ_ಅಂತ್ಯ // GGML_RPC_RDMA
}

ತಾರ್ಕಿಕ socket_t::impl::flush() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_RPC_RDMA_APPLE
    ಆದರೆ (use_rdma) {
        ಹಿಂತಿರುಗಿ rdma->flush();
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = ಪೂರ್ವನಿಯೋಜಿತ;

ತಾರ್ಕಿಕ socket_t::send_ದತ್ತಾಂಶ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    ಹಿಂತಿರುಗಿ pimpl->send_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ, size);
}

ತಾರ್ಕಿಕ socket_t::recv_ದತ್ತಾಂಶ(ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t size) {
    ಹಿಂತಿರುಗಿ pimpl->recv_ದತ್ತಾಂಶ(ದತ್ತಾಂಶ, size);
}

ತಾರ್ಕಿಕ socket_t::flush() {
    ಹಿಂತಿರುಗಿ pimpl->flush();
}

ಶೂನ್ಯ socket_t::get_caps(uint8_t * local_caps) {
    ಹಿಂತಿರುಗಿ pimpl->get_caps(local_caps);
}

ಶೂನ್ಯ socket_t::update_caps(ಸ್ಥಿರಾಂಕ uint8_t * remote_caps) {
    ಹಿಂತಿರುಗಿ pimpl->update_caps(remote_caps);
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * socket_t::transport_name() ಸ್ಥಿರಾಂಕ {
    ಹಿಂತಿರುಗಿ pimpl->use_rdma ? "RDMA" : "TCP";
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_valid_fd(sockfd_t sockfd) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಹಿಂತಿರುಗಿ sockfd != INVALID_SOCKET;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ sockfd >= 0;
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ಗಣ_no_delay(sockfd_t sockfd) {
    ಪೂರ್ಣಾಂಕ flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    ಪೂರ್ಣಾಂಕ ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (ಅಕ್ಷರ *)&flag, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಹಿಂತಿರುಗಿ ret == 0;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ಗಣ_reuse_addr(sockfd_t sockfd) {
    ಪೂರ್ಣಾಂಕ flag = 1;
    ಪೂರ್ಣಾಂಕ ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (ಅಕ್ಷರ *)&flag, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಹಿಂತಿರುಗಿ ret == 0;
}

socket_ptr socket_t::accept() {
    ಸ್ವಯಂಚಾಲಿತ client_socket_fd = ::accept(pimpl->fd, ಖಾಲಿ, ಖಾಲಿ);
    ಆದರೆ (!is_valid_fd(client_socket_fd)) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (!ಗಣ_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    ಹಿಂತಿರುಗಿ socket_ptr(ಹೊಸ socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * host, ಪೂರ್ಣಾಂಕ ಪೋರ್ಟ್) {
    ಸ್ವಯಂಚಾಲಿತ sockfd = socket(AF_INET, SOCK_ಹರಿವು, 0);
    ಆದರೆ (!is_valid_fd(sockfd)) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (!ಗಣ_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ರಚನೆ sockaddr_ಒಳ serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_ಪೋರ್ಟ್ = htons(ಪೋರ್ಟ್);

    ಆದರೆ (bind(sockfd, (ರಚನೆ sockaddr *) &serv_addr, ಗಾತ್ರ(serv_addr)) < 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (listen(sockfd, 1) < 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, ಪೋರ್ಟ್);
    ಹಿಂತಿರುಗಿ socket_ptr(ಹೊಸ socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * host, ಪೂರ್ಣಾಂಕ ಪೋರ್ಟ್) {
    ಸ್ವಯಂಚಾಲಿತ sockfd = socket(AF_INET, SOCK_ಹರಿವು, 0);
    ಆದರೆ (!is_valid_fd(sockfd)) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (!ಗಣ_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ರಚನೆ sockaddr_ಒಳ addr;
    addr.sin_family = AF_INET;
    addr.sin_ಪೋರ್ಟ್ = htons(ಪೋರ್ಟ್);
    ರಚನೆ hostent * server = gethostbyname(host);
    ಆದರೆ (server == ಖಾಲಿ) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಸ್ಮರಣೆ_ನಕಲು(&addr.sin_addr.s_addr, server->h_addr, server->h_ಉದ್ದಳತೆ);
    ಆದರೆ (::connect(sockfd, (ರಚನೆ sockaddr *)&addr, ಗಾತ್ರ(addr)) < 0) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, ಪೋರ್ಟ್);
    ಹಿಂತಿರುಗಿ socket_ptr(ಹೊಸ socket_t(std::make_unique<impl>(sockfd)));
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
ಸ್ಥಿರ std::mutex g_rpc_transport_mu;
ಸ್ಥಿರ ತಾರ್ಕಿಕ g_rpc_transport_wsa_started = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

ತಾರ್ಕಿಕ rpc_transport_init() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    std::lock_guard<std::mutex> ಬೀಗ(g_rpc_transport_mu);
    ಆದರೆ (g_rpc_transport_wsa_started) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    WSADATA wsa_ದತ್ತಾಂಶ;
    ಪೂರ್ಣಾಂಕ res = WSAStartup(MAKEWORD(2, 2), &wsa_ದತ್ತಾಂಶ);
    ಆದರೆ (res != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    g_rpc_transport_wsa_started = ನಿಜ;
    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ rpc_transport_shutdown() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    std::lock_guard<std::mutex> ಬೀಗ(g_rpc_transport_mu);
    ಆದರೆ (!g_rpc_transport_wsa_started) {
        ಹಿಂತಿರುಗಿ;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
}

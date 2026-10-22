#ਅਨ੍ਤਰ੍ਭਾਵ "transport.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "log.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#  define WIN32_LEAN_AND_ਔਸਤ
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#ਨਹੀਂਤਾਂ
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA
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
#ਜੇ_ਅੰਤ // GGML_RPC_RDMA

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
ਕਿਸਮਪਰਿਭਾਸ਼ਾ SOCKET sockfd_t;
ਵਰਤੋਂ ssize_t = __int64;
#ਨਹੀਂਤਾਂ
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ sockfd_t;
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA
ਸਥਿਰ constexpr size_t RDMA_GID_ਮਾਪ = 16;            // RoCE GID / IB GID is always 16 bytes
ਵਰਤੋਂ rdma_gid_t = std::array<uint8_t, RDMA_GID_ਮਾਪ>;
#ਜੇ_ਅੰਤ // GGML_RPC_RDMA

#ਜੇ defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
ਸਥਿਰ constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
ਸਥਿਰ constexpr ਪੂਰਨ    RDMA_RX_ਡੂੰਘਾਈ = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
ਸਥਿਰ constexpr ਸਵੈਚਾਲਿਤ   RDMA_SPIN_ਸਮਾਂ = std::chrono::milliseconds(100);

ਬਣਤਰ rdma_conn {
    ਬਣਤਰ ibv_context * ctx = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ibv_pd * pd  = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ibv_cq * scq = ਖਾਲੀ_ਸੂਚਕ;   // send completions
    ਬਣਤਰ ibv_cq * rcq = ਖਾਲੀ_ਸੂਚਕ;   // recv completions
    ਬਣਤਰ ibv_qp * qp  = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ibv_comp_channel * ch = ਖਾਲੀ_ਸੂਚਕ; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point ਆਖਰੀ_ਸਰਗਰਮ; // last completion or posted send

    ਖਾਲੀ          * tx_buf = ਖਾਲੀ_ਸੂਚਕ;
    ਬਣਤਰ ibv_mr * tx_mr  = ਖਾਲੀ_ਸੂਚਕ;

    ਖਾਲੀ          * rx_buf = ਖਾਲੀ_ਸੂਚਕ; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    ਬਣਤਰ ibv_mr * rx_mr  = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ             rx_ਸਿਰ = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_ਥਾਂ(ਪੂਰਨ i) ਅਚਲ {
        ਵਾਪਸ static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    ਤਾਰਕਿਕ post_rx(ਪੂਰਨ i) {
        ਬਣਤਰ ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_ਥਾਂ(i);
        sge.ਲੰਮਾਈ = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        ਬਣਤਰ ibv_recv_wr wr = {}, * bad = ਖਾਲੀ_ਸੂਚਕ;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.ਅੰਕ_sge = 1;
        ਵਾਪਸ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        ਜੇ (tx_mr) ibv_dereg_mr(tx_mr);
        ਜੇ (rx_mr) ibv_dereg_mr(rx_mr);
        ਮੁਕਤ(tx_buf);
        ਮੁਕਤ(rx_buf);
        ਜੇ (qp)  ibv_destroy_qp(qp);
        ਜੇ (scq) ibv_destroy_cq(scq);
        ਜੇ (rcq) ibv_destroy_cq(rcq);
        ਜੇ (ch)  ibv_destroy_comp_channel(ch);
        ਜੇ (pd)  ibv_dealloc_pd(pd);
        ਜੇ (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
ਬਣਤਰ rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_ਮਾਪ] = {};
    uint8_t  ib_ਪੋਰਟ = 0;
    ਪੂਰਨ      gid_idx = 0;
    ਗਿਣਤੀ ibv_mtu ਰਸਤਾ_mtu = IBV_MTU_1024;
};

ਬਣਤਰ rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_ਮਾਪ];
};

static_assert(ਆਕਾਰ(rdma_caps) == RPC_CONN_CAPS_ਮਾਪ, "rdma_caps must match conn_caps size");

#ਜੇ_ਅੰਤ // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

ਬਣਤਰ socket_t::impl {
    impl(sockfd_t fd) : use_rdma(ਝੂਠ), fd(fd) {}
    ~impl();
    ਤਾਰਕਿਕ send_ਅੰਕੜੇ(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size);
    ਤਾਰਕਿਕ recv_ਅੰਕੜੇ(ਖਾਲੀ * ਅੰਕੜੇ, size_t size);
    ਤਾਰਕਿਕ flush();
    ਖਾਲੀ get_caps(uint8_t * local_caps);
    ਖਾਲੀ update_caps(ਅਚਲ uint8_t * remote_caps);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_ਨਿਸ਼ਾਨਾ_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  ਨਹੀਂਤਾਂ
    ਤਾਰਕਿਕ rdma_probe();
    ਤਾਰਕਿਕ rdma_send(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size);
    ਤਾਰਕਿਕ rdma_recv(ਖਾਲੀ * ਅੰਕੜੇ, size_t size);
    ਤਾਰਕਿਕ tcp_peer_closed();
    ਤਾਰਕਿਕ rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ਅਚਲ uint8_t * remote_gid);
    ਤਾਰਕਿਕ rdma_poll(ਬਣਤਰ ibv_cq * cq, ਬਣਤਰ ibv_wc * wc);
    ਤਾਰਕਿਕ rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#ਜੇ_ਅੰਤ // GGML_RPC_RDMA
    ਤਾਰਕਿਕ     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA
    rdma.reset();
#ਜੇ_ਅੰਤ // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, ਇਹ->fd);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਜੇ (fd != INVALID_SOCKET) closesocket(ਇਹ->fd);
#ਨਹੀਂਤਾਂ
    ਜੇ (fd >= 0) close(ਇਹ->fd);
#ਜੇ_ਅੰਤ
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA

// Build a RoCE GID-shaped 16-byte target from a TCP socket's local address.
// Used to match the socket's local IP against the kernel's GID table so that
// a single memcmp handles IPv4, IPv4-mapped IPv6, and native IPv6 uniformly:
//   AF_INET                -> ::ffff:a.b.c.d  (bytes 10-11 = 0xff, last 4 = IPv4)
//   AF_INET6 (IPv4-mapped) -> ::ffff:a.b.c.d  (already in GID shape)
//   AF_INET6 (native v6)   -> the 16-byte IPv6 address as-is
// Returns std::nullopt on unsupported family or getsockname failure.
std::optional<rdma_gid_t> socket_t::impl::rdma_build_ਨਿਸ਼ਾਨਾ_gid() {
    sockaddr_storage addr = {};
    socklen_t addr_len = ਆਕਾਰ(addr);
    ਜੇ (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        ਵਾਪਸ std::nullopt;
    }
    rdma_gid_t ਨਿਸ਼ਾਨਾ = {};
    ਜੇ (addr.ss_family == AF_INET) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * a = reinterpret_cast<ਅਚਲ sockaddr_ਵਿੱਚ *>(&addr);
        ਨਿਸ਼ਾਨਾ[10] = 0xff;
        ਨਿਸ਼ਾਨਾ[11] = 0xff;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਨਿਸ਼ਾਨਾ[12], &a->sin_addr, 4);
        ਵਾਪਸ ਨਿਸ਼ਾਨਾ;
    }
    ਜੇ (addr.ss_family == AF_INET6) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ * a = reinterpret_cast<ਅਚਲ sockaddr_in6 *>(&addr);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਨਿਸ਼ਾਨਾ.ਅੰਕੜੇ(), &a->sin6_addr, RDMA_GID_ਮਾਪ);
        ਵਾਪਸ ਨਿਸ਼ਾਨਾ;
    }
    ਵਾਪਸ std::nullopt;
}

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA_APPLE

ਤਾਰਕਿਕ socket_t::impl::tcp_peer_closed() {
    ਜੇ (fd < 0) ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਬਣਤਰ pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    ਪੂਰਨ r = poll(&pfd, 1, 0);
    ਵਾਪਸ r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
}

ਤਾਰਕਿਕ socket_t::impl::rdma_probe() {
    ਅਚਲ ਅੱਖਰ * dev_env = std::getenv("GGML_RDMA_DEV");
    ਅਚਲ ਅੱਖਰ * gid_env = std::getenv("GGML_RDMA_GID");

    ਸਵੈਚਾਲਿਤ ਨਿਸ਼ਾਨਾ_gid = rdma_build_ਨਿਸ਼ਾਨਾ_gid();
    ਜੇ (!ਨਿਸ਼ਾਨਾ_gid) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ uint8_t ib_ਪੋਰਟ = 1;
    ਪੂਰਨ ਅੰਕ_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&ਅੰਕ_devs);
    ਜੇ (!devs || ਅੰਕ_devs == 0) ਵਾਪਸ ਝੂਠ;

    ibv_context * ibctx = ਖਾਲੀ_ਸੂਚਕ;
    ਅਚਲ ਅੱਖਰ * matched_dev = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ gid_idx = gid_env ? atoi(gid_env) : -1;
    ਪੂਰਨ gid_ਸੰਸਕਰਣ = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    ਲਈ (ਪੂਰਨ d = 0; d < ਅੰਕ_devs; d++) {
        ਅਚਲ ਅੱਖਰ * dn = ibv_get_device_name(devs[d]);
        ਜੇ (dev_env && strcmp(dev_env, dn) != 0) ਜਾਰੀ;

        ibv_context * ctx = ibv_open_device(devs[d]);
        ਜੇ (!ctx) ਜਾਰੀ;

        ibv_ਪੋਰਟ_attr pa;
        ਜੇ (ibv_query_ਪੋਰਟ(ctx, ib_ਪੋਰਟ, &pa) != 0) { ibv_close_device(ctx); ਜਾਰੀ; }

        ਪੂਰਨ ਮਿਲਿਆ_gid = gid_idx;
        ਪੂਰਨ ਮਿਲਿਆ_ਸੰਸਕਰਣ = IBV_GID_TYPE_IB;
        ਜੇ (ਮਿਲਿਆ_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            ਪੂਰਨ v2_idx = -1;
            ਪੂਰਨ v1_idx = -1;
            ਲਈ (ਪੂਰਨ i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_ਇੰਦਰਾਜ ਇੰਦਰਾਜ = {};
                ਜੇ (ibv_query_gid_ex(ctx, ib_ਪੋਰਟ, i, &ਇੰਦਰਾਜ, 0) != 0) ਜਾਰੀ;
                ਜੇ (memcmp(ਇੰਦਰਾਜ.gid.raw, ਨਿਸ਼ਾਨਾ_gid->ਅੰਕੜੇ(), RDMA_GID_ਮਾਪ) != 0) ਜਾਰੀ;
                ਜੇ (ਇੰਦਰਾਜ.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } ਨਹੀਂਤਾਂ ਜੇ (ਇੰਦਰਾਜ.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            ਜੇ (v2_idx >= 0) {
                ਮਿਲਿਆ_gid = v2_idx;
                ਮਿਲਿਆ_ਸੰਸਕਰਣ = IBV_GID_TYPE_ROCE_V2;
            } ਨਹੀਂਤਾਂ ਜੇ (v1_idx >= 0) {
                ਮਿਲਿਆ_gid = v1_idx;
                ਮਿਲਿਆ_ਸੰਸਕਰਣ = IBV_GID_TYPE_ROCE_V1;
            }
        } ਨਹੀਂਤਾਂ {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_ਇੰਦਰਾਜ ਇੰਦਰਾਜ = {};
            ਜੇ (ibv_query_gid_ex(ctx, ib_ਪੋਰਟ, ਮਿਲਿਆ_gid, &ਇੰਦਰਾਜ, 0) == 0) {
                ਮਿਲਿਆ_ਸੰਸਕਰਣ = ਇੰਦਰਾਜ.gid_type;
            }
        }
        ਜੇ (ਮਿਲਿਆ_gid >= 0) {
            ibctx = ctx;
            gid_idx = ਮਿਲਿਆ_gid;
            gid_ਸੰਸਕਰਣ = ਮਿਲਿਆ_ਸੰਸਕਰਣ;
            matched_dev = dn;
            rdma_local.ਰਸਤਾ_mtu = pa.ਸਰਗਰਮ_mtu;
            ਰੋਕੋ;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    ਜੇ (!ibctx) ਵਾਪਸ ਝੂਠ;

    rdma_local.ib_ਪੋਰਟ = ib_ਪੋਰਟ;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    ਜੇ (!rdma->pd) ਵਾਪਸ ਝੂਠ;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, ਖਾਲੀ_ਸੂਚਕ, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_ਡੂੰਘਾਈ + 4, ਖਾਲੀ_ਸੂਚਕ, rdma->ch, 0);
    ਜੇ (!rdma->scq || !rdma->rcq) ਵਾਪਸ ਝੂਠ;

    ibv_qp_init_attr qia = {};
    qia.send_cq = rdma->scq;
    qia.recv_cq = rdma->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr     = 4;
    qia.cap.max_recv_wr     = RDMA_RX_ਡੂੰਘਾਈ + 4;
    qia.cap.max_send_sge    = 1;
    qia.cap.max_recv_sge    = 1;
    qia.cap.max_inline_ਅੰਕੜੇ = 256;

    rdma->qp = ibv_create_qp(rdma->pd, &qia);
    ਜੇ (!rdma->qp) ਵਾਪਸ ਝੂਠ;
    rdma->max_inline = qia.cap.max_inline_ਅੰਕੜੇ;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_ਡੂੰਘਾਈ) * RDMA_CHUNK);
    ਜੇ (!rdma->tx_buf || !rdma->rx_buf) ਵਾਪਸ ਝੂਠ;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_ਡੂੰਘਾਈ) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    ਜੇ (!rdma->tx_mr || !rdma->rx_mr) ਵਾਪਸ ਝੂਠ;

    ibv_gid local_gid;
    ਜੇ (ibv_query_gid(ibctx, ib_ਪੋਰਟ, gid_idx, &local_gid) != 0) ਵਾਪਸ ਝੂਠ;

    rdma_local.qpn = rdma->qp->qp_ਅੰਕ;
    rdma_local.psn = rdma->qp->qp_ਅੰਕ & 0xffffff;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rdma_local.gid, &local_gid, RDMA_GID_ਮਾਪ);

    ਅਚਲ ਅੱਖਰ * ver_str = "";
    ਜੇ (gid_ਸੰਸਕਰਣ == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } ਨਹੀਂਤਾਂ ਜੇ (gid_ਸੰਸਕਰਣ == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    ਵਾਪਸ ਸੱਚ;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
ਤਾਰਕਿਕ socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, ਅਚਲ uint8_t * remote_gid) {
    // RESET -> INIT
    {
        ਬਣਤਰ ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.ਪੋਰਟ_ਅੰਕ        = rdma_local.ib_ਪੋਰਟ;
        a.pkey_ਸੂਚਕਾਂਕ      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        ਜੇ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_ਸੂਚਕਾਂਕ | IBV_QP_ਪੋਰਟ | IBV_QP_ACCESS_FLAGS) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < RDMA_RX_ਡੂੰਘਾਈ; i++) {
        ਜੇ (!rdma->post_rx(i)) ਵਾਪਸ ਝੂਠ;
    }

    // INIT -> RTR
    {
        ਬਣਤਰ ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.ਰਸਤਾ_mtu           = rdma_local.ਰਸਤਾ_mtu;
        a.dest_qp_ਅੰਕ        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_ਮਾਪ);
        a.ah_attr.grh.hop_ਹੱਦਬੰਦੀ  = 1;
        a.ah_attr.grh.sgid_ਸੂਚਕਾਂਕ = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.ਪੋਰਟ_ਅੰਕ   = rdma_local.ib_ਪੋਰਟ;
        ਜੇ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_ਰਸਤਾ_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    // RTR -> RTS
    {
        ਬਣਤਰ ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        ਜੇ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    rdma->ਆਖਰੀ_ਸਰਗਰਮ = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.ਰਸਤਾ_mtu, RDMA_RX_ਡੂੰਘਾਈ);
    ਵਾਪਸ ਸੱਚ;
}

// Sleep until the completion channel has an event or the TCP peer closes.
ਤਾਰਕਿਕ socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    ਬਣਤਰ pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    ਜੇ (poll(pfds, 2, -1) < 0) {
        ਵਾਪਸ errno == EINTR;
    }
    ਜੇ (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (pfds[0].revents & POLLIN) {
        ਬਣਤਰ ibv_cq * ev_cq  = ਖਾਲੀ_ਸੂਚਕ;
        ਖਾਲੀ          * ev_ctx = ਖਾਲੀ_ਸੂਚਕ;
        ਜੇ (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            ਵਾਪਸ ਝੂਠ;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ socket_t::impl::rdma_poll(ਬਣਤਰ ibv_cq * cq, ਬਣਤਰ ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    ਤਾਰਕਿਕ armed = ਝੂਠ;
    ਲਈ (uint64_t s = 1; ; s++) {
        ਪੂਰਨ n = ibv_poll_cq(cq, 1, wc);
        ਜੇ (n > 0) {
            c->ਆਖਰੀ_ਸਰਗਰਮ = std::chrono::steady_clock::now();
            ਜੇ (wc->ਸਥਿਤੀ != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->ਸਥਿਤੀ, ibv_wc_ਸਥਿਤੀ_str(wc->ਸਥਿਤੀ), wc->vendor_err);
            }
            ਵਾਪਸ wc->ਸਥਿਤੀ == IBV_WC_SUCCESS;
        }
        ਜੇ (n < 0) ਵਾਪਸ ਝੂਠ;
        ਜੇ (armed) {
            // armed and still empty: sleep until the next completion
            ਜੇ (!rdma_wait_event()) {
                ਵਾਪਸ ਝੂਠ;
            }
            armed = ਝੂਠ;
            ਜਾਰੀ;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        ਜੇ (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->ਆਖਰੀ_ਸਰਗਰਮ > RDMA_SPIN_ਸਮਾਂ) {
            ਜੇ (ibv_req_notify_cq(cq, 0) != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            armed = ਸੱਚ;
            ਜਾਰੀ;
        }
        ਜੇ ((s & 0xFFFFF) == 0) {
            ਜੇ (tcp_peer_closed()) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }
}

ਤਾਰਕਿਕ socket_t::impl::rdma_send(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    rdma_conn * c = rdma.get();
    ਅਚਲ uint8_t * src = (ਅਚਲ uint8_t *)ਅੰਕੜੇ;
    size_t rem = size;
    ਜਦੋਂ (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        ਬਣਤਰ ibv_sge sge = {};
        ਬਣਤਰ ibv_send_wr wr = {}, * bad = ਖਾਲੀ_ਸੂਚਕ;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.ਅੰਕ_sge = 1;

        ਜੇ (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.ਲੰਮਾਈ = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } ਨਹੀਂਤਾਂ {
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.ਲੰਮਾਈ = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        ਜੇ (ibv_post_send(c->qp, &wr, &bad) != 0) ਵਾਪਸ ਝੂਠ;
        c->ਆਖਰੀ_ਸਰਗਰਮ = std::chrono::steady_clock::now();
        ਬਣਤਰ ibv_wc wc;
        ਜੇ (!rdma_poll(c->scq, &wc)) ਵਾਪਸ ਝੂਠ;

        src += chunk;
        rem -= chunk;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ socket_t::impl::rdma_recv(ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)ਅੰਕੜੇ;
    size_t rem = size;
    ਜਦੋਂ (rem > 0) {
        ਬਣਤਰ ibv_wc wc;
        ਜੇ (!rdma_poll(c->rcq, &wc)) ਵਾਪਸ ਝੂਠ;

        ਪੂਰਨ ਥਾਂ = (ਪੂਰਨ)wc.wr_id;
        size_t got = wc.byte_len;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst, c->rx_ਥਾਂ(ਥਾਂ), got);

        ਜੇ (!c->post_rx(ਥਾਂ)) ਵਾਪਸ ਝੂਠ;

        dst += got;
        rem -= got;
    }
    ਵਾਪਸ ਸੱਚ;
}

#ਜੇ_ਅੰਤ // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#ਜੇ_ਅੰਤ // GGML_RPC_RDMA

ਤਾਰਕਿਕ socket_t::impl::send_ਅੰਕੜੇ(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_ਮਾਪ(size).c_str());
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA_APPLE
    ਜੇ (use_rdma) {
        ਵਾਪਸ rdma->send(ਅੰਕੜੇ, size);
    }
#ਜਾਂਜੇ defined(GGML_RPC_RDMA)
    ਜੇ (use_rdma) {
        ਵਾਪਸ rdma_send(ਅੰਕੜੇ, size);
    }
#ਜੇ_ਅੰਤ
    size_t ਬਾਈਟਾਂ_ਭੇਜਿਆ = 0;
    ਜਦੋਂ (ਬਾਈਟਾਂ_ਭੇਜਿਆ < size) {
        size_t ਮਾਪ_ਮੰਜ਼ਿਲ_send = std::min(size - ਬਾਈਟਾਂ_ਭੇਜਿਆ, MAX_CHUNK_ਮਾਪ);
        ssize_t n = send(fd, (ਅਚਲ ਅੱਖਰ *)ਅੰਕੜੇ + ਬਾਈਟਾਂ_ਭੇਜਿਆ, ਮਾਪ_ਮੰਜ਼ਿਲ_send, 0);
        ਜੇ (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           ਬਾਈਟਾਂ_ਭੇਜਿਆ, ਮਾਪ_ਮੰਜ਼ਿਲ_send);
            ਵਾਪਸ ਝੂਠ;
        }
        ਬਾਈਟਾਂ_ਭੇਜਿਆ += (size_t)n;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ socket_t::impl::recv_ਅੰਕੜੇ(ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_ਮਾਪ(size).c_str());
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA_APPLE
    ਜੇ (use_rdma) {
        ਵਾਪਸ rdma->recv(ਅੰਕੜੇ, size);
    }
#ਜਾਂਜੇ defined(GGML_RPC_RDMA)
    ਜੇ (use_rdma) {
        ਵਾਪਸ rdma_recv(ਅੰਕੜੇ, size);
    }
#ਜੇ_ਅੰਤ
    size_t ਬਾਈਟਾਂ_recv = 0;
    ਜਦੋਂ (ਬਾਈਟਾਂ_recv < size) {
        size_t ਮਾਪ_ਮੰਜ਼ਿਲ_recv = std::min(size - ਬਾਈਟਾਂ_recv, MAX_CHUNK_ਮਾਪ);
        ssize_t n = recv(fd, (ਅੱਖਰ *)ਅੰਕੜੇ + ਬਾਈਟਾਂ_recv, ਮਾਪ_ਮੰਜ਼ਿਲ_recv, 0);
        ਜੇ (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           ਬਾਈਟਾਂ_recv, ਮਾਪ_ਮੰਜ਼ਿਲ_recv);
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            ਵਾਪਸ ਝੂਠ;
        }
        ਬਾਈਟਾਂ_recv += (size_t)n;
    }
    ਵਾਪਸ ਸੱਚ;
}

ਖਾਲੀ socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_ਮਾਪ);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA
    ਜੇ (std::getenv("GGML_RPC_NO_RDMA")) {
        ਵਾਪਸ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    ਸਵੈਚਾਲਿਤ ਨਿਸ਼ਾਨਾ_gid = rdma_build_ਨਿਸ਼ਾਨਾ_gid();
    ਜੇ (ਨਿਸ਼ਾਨਾ_gid) {
        rdma = apple_rdma::probe(fd, ਨਿਸ਼ਾਨਾ_gid->ਅੰਕੜੇ(), local_caps);
    }
    ਜੇ (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  ਨਹੀਂਤਾਂ
    rdma_local = {};
    ਜੇ (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(rc.gid, rdma_local.gid, RDMA_GID_ਮਾਪ);
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(local_caps, &rc, ਆਕਾਰ(rc));
    } ਨਹੀਂਤਾਂ {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#ਜੇ_ਅੰਤ // GGML_RPC_RDMA
}

ਖਾਲੀ socket_t::impl::update_caps(ਅਚਲ uint8_t * remote_caps) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    ਤਾਰਕਿਕ remote_rdma = ਝੂਠ;
    ਲਈ (size_t i = 0; i < RPC_CONN_CAPS_ਮਾਪ; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    ਜੇ (!rdma || !remote_rdma) {
        ਜੇ (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        ਵਾਪਸ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    ਤਾਰਕਿਕ activated = rdma->activate(remote_caps);
#  ਨਹੀਂਤਾਂ
    rdma_caps rc = {};
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rc, remote_caps, ਆਕਾਰ(rc));
    ਤਾਰਕਿਕ activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    ਜੇ (activated) {
        use_rdma = ਸੱਚ;
    } ਨਹੀਂਤਾਂ {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#ਨਹੀਂਤਾਂ
    (ਖਾਲੀ)remote_caps;
#ਜੇ_ਅੰਤ // GGML_RPC_RDMA
}

ਤਾਰਕਿਕ socket_t::impl::flush() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_RPC_RDMA_APPLE
    ਜੇ (use_rdma) {
        ਵਾਪਸ rdma->flush();
    }
#ਜੇ_ਅੰਤ
    ਵਾਪਸ ਸੱਚ;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = ਮੂਲ;

ਤਾਰਕਿਕ socket_t::send_ਅੰਕੜੇ(ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    ਵਾਪਸ pimpl->send_ਅੰਕੜੇ(ਅੰਕੜੇ, size);
}

ਤਾਰਕਿਕ socket_t::recv_ਅੰਕੜੇ(ਖਾਲੀ * ਅੰਕੜੇ, size_t size) {
    ਵਾਪਸ pimpl->recv_ਅੰਕੜੇ(ਅੰਕੜੇ, size);
}

ਤਾਰਕਿਕ socket_t::flush() {
    ਵਾਪਸ pimpl->flush();
}

ਖਾਲੀ socket_t::get_caps(uint8_t * local_caps) {
    ਵਾਪਸ pimpl->get_caps(local_caps);
}

ਖਾਲੀ socket_t::update_caps(ਅਚਲ uint8_t * remote_caps) {
    ਵਾਪਸ pimpl->update_caps(remote_caps);
}

ਅਚਲ ਅੱਖਰ * socket_t::transport_name() ਅਚਲ {
    ਵਾਪਸ pimpl->use_rdma ? "RDMA" : "TCP";
}

ਸਥਿਰ ਤਾਰਕਿਕ is_valid_fd(sockfd_t sockfd) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਵਾਪਸ sockfd != INVALID_SOCKET;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ sockfd >= 0;
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਤਾਰਕਿਕ ਸੈੱਟ_no_delay(sockfd_t sockfd) {
    ਪੂਰਨ flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    ਪੂਰਨ ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (ਅੱਖਰ *)&flag, ਆਕਾਰ(ਪੂਰਨ));
    ਵਾਪਸ ret == 0;
}

ਸਥਿਰ ਤਾਰਕਿਕ ਸੈੱਟ_reuse_addr(sockfd_t sockfd) {
    ਪੂਰਨ flag = 1;
    ਪੂਰਨ ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (ਅੱਖਰ *)&flag, ਆਕਾਰ(ਪੂਰਨ));
    ਵਾਪਸ ret == 0;
}

socket_ptr socket_t::accept() {
    ਸਵੈਚਾਲਿਤ client_socket_fd = ::accept(pimpl->fd, ਨਿਰਾ, ਨਿਰਾ);
    ਜੇ (!is_valid_fd(client_socket_fd)) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (!ਸੈੱਟ_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    ਵਾਪਸ socket_ptr(ਨਵਾਂ socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(ਅਚਲ ਅੱਖਰ * host, ਪੂਰਨ ਪੋਰਟ) {
    ਸਵੈਚਾਲਿਤ sockfd = socket(AF_INET, SOCK_ਧਾਰਾ, 0);
    ਜੇ (!is_valid_fd(sockfd)) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (!ਸੈੱਟ_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਬਣਤਰ sockaddr_ਵਿੱਚ serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_ਪੋਰਟ = htons(ਪੋਰਟ);

    ਜੇ (bind(sockfd, (ਬਣਤਰ sockaddr *) &serv_addr, ਆਕਾਰ(serv_addr)) < 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (listen(sockfd, 1) < 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, ਪੋਰਟ);
    ਵਾਪਸ socket_ptr(ਨਵਾਂ socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(ਅਚਲ ਅੱਖਰ * host, ਪੂਰਨ ਪੋਰਟ) {
    ਸਵੈਚਾਲਿਤ sockfd = socket(AF_INET, SOCK_ਧਾਰਾ, 0);
    ਜੇ (!is_valid_fd(sockfd)) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (!ਸੈੱਟ_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਬਣਤਰ sockaddr_ਵਿੱਚ addr;
    addr.sin_family = AF_INET;
    addr.sin_ਪੋਰਟ = htons(ਪੋਰਟ);
    ਬਣਤਰ hostent * server = gethostbyname(host);
    ਜੇ (server == ਨਿਰਾ) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&addr.sin_addr.s_addr, server->h_addr, server->h_ਲੰਮਾਈ);
    ਜੇ (::connect(sockfd, (ਬਣਤਰ sockaddr *)&addr, ਆਕਾਰ(addr)) < 0) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, ਪੋਰਟ);
    ਵਾਪਸ socket_ptr(ਨਵਾਂ socket_t(std::make_unique<impl>(sockfd)));
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
ਸਥਿਰ std::mutex g_rpc_transport_mu;
ਸਥਿਰ ਤਾਰਕਿਕ g_rpc_transport_wsa_started = ਝੂਠ;
#ਜੇ_ਅੰਤ

ਤਾਰਕਿਕ rpc_transport_init() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    std::lock_guard<std::mutex> ਤਾਲਾ(g_rpc_transport_mu);
    ਜੇ (g_rpc_transport_wsa_started) {
        ਵਾਪਸ ਸੱਚ;
    }
    WSADATA wsa_ਅੰਕੜੇ;
    ਪੂਰਨ res = WSAStartup(MAKEWORD(2, 2), &wsa_ਅੰਕੜੇ);
    ਜੇ (res != 0) {
        ਵਾਪਸ ਝੂਠ;
    }
    g_rpc_transport_wsa_started = ਸੱਚ;
    ਵਾਪਸ ਸੱਚ;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ ਸੱਚ;
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ rpc_transport_shutdown() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    std::lock_guard<std::mutex> ਤਾਲਾ(g_rpc_transport_mu);
    ਜੇ (!g_rpc_transport_wsa_started) {
        ਵਾਪਸ;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = ਝੂਠ;
#ਜੇ_ਅੰਤ
}

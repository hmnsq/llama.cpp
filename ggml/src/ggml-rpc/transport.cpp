#അന്തര്ഭാവ "transport.h"
#അന്തര്ഭാവ "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#അല്ലെങ്കിൽ
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <optional>

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
തരനിർവചനം SOCKET sockfd_t;
using ssize_t = __int64;
#അല്ലെങ്കിൽ
തരനിർവചനം പൂർണ്ണസംഖ്യ sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
സ്ഥിതം constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#എങ്കിൽ defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
സ്ഥിതം constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
സ്ഥിതം constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

ഘടന rdma_conn {
    ഘടന ibv_context * ctx = nullptr;
    ഘടന ibv_pd * pd  = nullptr;
    ഘടന ibv_cq * scq = nullptr;   // send completions
    ഘടന ibv_cq * rcq = nullptr;   // recv completions
    ഘടന ibv_qp * qp  = nullptr;
    ഘടന ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    ശൂന്യം          * tx_buf = nullptr;
    ഘടന ibv_mr * tx_mr  = nullptr;

    ശൂന്യം          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    ഘടന ibv_mr * rx_mr  = nullptr;
    പൂർണ്ണസംഖ്യ             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(പൂർണ്ണസംഖ്യ i) സ്ഥിരം {
        തിരികെ static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    തർക്കം post_rx(പൂർണ്ണസംഖ്യ i) {
        ഘടന ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        ഘടന ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.അക്കം_sge = 1;
        തിരികെ ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        എങ്കിൽ (tx_mr) ibv_dereg_mr(tx_mr);
        എങ്കിൽ (rx_mr) ibv_dereg_mr(rx_mr);
        സ്വതന്ത്രം(tx_buf);
        സ്വതന്ത്രം(rx_buf);
        എങ്കിൽ (qp)  ibv_destroy_qp(qp);
        എങ്കിൽ (scq) ibv_destroy_cq(scq);
        എങ്കിൽ (rcq) ibv_destroy_cq(rcq);
        എങ്കിൽ (ch)  ibv_destroy_comp_channel(ch);
        എങ്കിൽ (pd)  ibv_dealloc_pd(pd);
        എങ്കിൽ (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
ഘടന rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    പൂർണ്ണസംഖ്യ      gid_idx = 0;
    എണ്ണൽ ibv_mtu path_mtu = IBV_MTU_1024;
};

ഘടന rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(വലുപ്പം(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

ഘടന socket_t::impl {
    impl(sockfd_t fd) : use_rdma(അസത്യം), fd(fd) {}
    ~impl();
    തർക്കം send_data(സ്ഥിരം ശൂന്യം * data, size_t size);
    തർക്കം recv_data(ശൂന്യം * data, size_t size);
    തർക്കം flush();
    ശൂന്യം get_caps(uint8_t * local_caps);
    ശൂന്യം update_caps(സ്ഥിരം uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  അല്ലെങ്കിൽ
    തർക്കം rdma_probe();
    തർക്കം rdma_send(സ്ഥിരം ശൂന്യം * data, size_t size);
    തർക്കം rdma_recv(ശൂന്യം * data, size_t size);
    തർക്കം tcp_peer_closed();
    തർക്കം rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, സ്ഥിരം uint8_t * remote_gid);
    തർക്കം rdma_poll(ഘടന ibv_cq * cq, ഘടന ibv_wc * wc);
    തർക്കം rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    തർക്കം     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, ഇത്->fd);
#ifdef _WIN32
    എങ്കിൽ (fd != INVALID_SOCKET) closesocket(ഇത്->fd);
#അല്ലെങ്കിൽ
    എങ്കിൽ (fd >= 0) close(ഇത്->fd);
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
    socklen_t addr_len = വലുപ്പം(addr);
    എങ്കിൽ (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        തിരികെ std::nullopt;
    }
    rdma_gid_t target = {};
    എങ്കിൽ (addr.ss_family == AF_INET) {
        സ്ഥിരം auto * a = reinterpret_cast<സ്ഥിരം sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        തിരികെ target;
    }
    എങ്കിൽ (addr.ss_family == AF_INET6) {
        സ്ഥിരം auto * a = reinterpret_cast<സ്ഥിരം sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        തിരികെ target;
    }
    തിരികെ std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

തർക്കം socket_t::impl::tcp_peer_closed() {
    എങ്കിൽ (fd < 0) തിരികെ അസത്യം;
#ifndef _WIN32
    ഘടന pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    പൂർണ്ണസംഖ്യ r = poll(&pfd, 1, 0);
    തിരികെ r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#അല്ലെങ്കിൽ
    തിരികെ അസത്യം;
#endif
}

തർക്കം socket_t::impl::rdma_probe() {
    സ്ഥിരം അക്ഷരം * dev_env = std::getenv("GGML_RDMA_DEV");
    സ്ഥിരം അക്ഷരം * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    എങ്കിൽ (!target_gid) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം uint8_t ib_port = 1;
    പൂർണ്ണസംഖ്യ അക്കം_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&അക്കം_devs);
    എങ്കിൽ (!devs || അക്കം_devs == 0) തിരികെ അസത്യം;

    ibv_context * ibctx = nullptr;
    സ്ഥിരം അക്ഷരം * matched_dev = nullptr;
    പൂർണ്ണസംഖ്യ gid_idx = gid_env ? atoi(gid_env) : -1;
    പൂർണ്ണസംഖ്യ gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    വേണ്ടി (പൂർണ്ണസംഖ്യ d = 0; d < അക്കം_devs; d++) {
        സ്ഥിരം അക്ഷരം * dn = ibv_get_device_name(devs[d]);
        എങ്കിൽ (dev_env && strcmp(dev_env, dn) != 0) തുടരുക;

        ibv_context * ctx = ibv_open_device(devs[d]);
        എങ്കിൽ (!ctx) തുടരുക;

        ibv_port_attr pa;
        എങ്കിൽ (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); തുടരുക; }

        പൂർണ്ണസംഖ്യ found_gid = gid_idx;
        പൂർണ്ണസംഖ്യ found_version = IBV_GID_TYPE_IB;
        എങ്കിൽ (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            പൂർണ്ണസംഖ്യ v2_idx = -1;
            പൂർണ്ണസംഖ്യ v1_idx = -1;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                എങ്കിൽ (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) തുടരുക;
                എങ്കിൽ (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) തുടരുക;
                എങ്കിൽ (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } അല്ലെങ്കിൽ എങ്കിൽ (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            എങ്കിൽ (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } അല്ലെങ്കിൽ എങ്കിൽ (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } അല്ലെങ്കിൽ {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            എങ്കിൽ (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        എങ്കിൽ (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            നിർത്തുക;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    എങ്കിൽ (!ibctx) തിരികെ അസത്യം;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    എങ്കിൽ (!rdma->pd) തിരികെ അസത്യം;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    എങ്കിൽ (!rdma->scq || !rdma->rcq) തിരികെ അസത്യം;

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
    എങ്കിൽ (!rdma->qp) തിരികെ അസത്യം;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    എങ്കിൽ (!rdma->tx_buf || !rdma->rx_buf) തിരികെ അസത്യം;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    എങ്കിൽ (!rdma->tx_mr || !rdma->rx_mr) തിരികെ അസത്യം;

    ibv_gid local_gid;
    എങ്കിൽ (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) തിരികെ അസത്യം;

    rdma_local.qpn = rdma->qp->qp_അക്കം;
    rdma_local.psn = rdma->qp->qp_അക്കം & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    സ്ഥിരം അക്ഷരം * ver_str = "";
    എങ്കിൽ (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } അല്ലെങ്കിൽ എങ്കിൽ (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    തിരികെ സത്യം;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
തർക്കം socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, സ്ഥിരം uint8_t * remote_gid) {
    // RESET -> INIT
    {
        ഘടന ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_അക്കം        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        എങ്കിൽ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            തിരികെ അസത്യം;
        }
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < RDMA_RX_DEPTH; i++) {
        എങ്കിൽ (!rdma->post_rx(i)) തിരികെ അസത്യം;
    }

    // INIT -> RTR
    {
        ഘടന ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_അക്കം        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_അക്കം   = rdma_local.ib_port;
        എങ്കിൽ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            തിരികെ അസത്യം;
        }
    }

    // RTR -> RTS
    {
        ഘടന ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        എങ്കിൽ (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            തിരികെ അസത്യം;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    തിരികെ സത്യം;
}

// Sleep until the completion channel has an event or the TCP peer closes.
തർക്കം socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    ഘടന pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    എങ്കിൽ (poll(pfds, 2, -1) < 0) {
        തിരികെ errno == EINTR;
    }
    എങ്കിൽ (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (pfds[0].revents & POLLIN) {
        ഘടന ibv_cq * ev_cq  = nullptr;
        ശൂന്യം          * ev_ctx = nullptr;
        എങ്കിൽ (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            തിരികെ അസത്യം;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    തിരികെ സത്യം;
}

തർക്കം socket_t::impl::rdma_poll(ഘടന ibv_cq * cq, ഘടന ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    തർക്കം armed = അസത്യം;
    വേണ്ടി (uint64_t s = 1; ; s++) {
        പൂർണ്ണസംഖ്യ n = ibv_poll_cq(cq, 1, wc);
        എങ്കിൽ (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            എങ്കിൽ (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            തിരികെ wc->status == IBV_WC_SUCCESS;
        }
        എങ്കിൽ (n < 0) തിരികെ അസത്യം;
        എങ്കിൽ (armed) {
            // armed and still empty: sleep until the next completion
            എങ്കിൽ (!rdma_wait_event()) {
                തിരികെ അസത്യം;
            }
            armed = അസത്യം;
            തുടരുക;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        എങ്കിൽ (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            എങ്കിൽ (ibv_req_notify_cq(cq, 0) != 0) {
                തിരികെ അസത്യം;
            }
            armed = സത്യം;
            തുടരുക;
        }
        എങ്കിൽ ((s & 0xFFFFF) == 0) {
            എങ്കിൽ (tcp_peer_closed()) {
                തിരികെ അസത്യം;
            }
        }
    }
}

തർക്കം socket_t::impl::rdma_send(സ്ഥിരം ശൂന്യം * data, size_t size) {
    rdma_conn * c = rdma.get();
    സ്ഥിരം uint8_t * src = (സ്ഥിരം uint8_t *)data;
    size_t rem = size;
    ആകുമ്പോൾ (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        ഘടന ibv_sge sge = {};
        ഘടന ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.അക്കം_sge = 1;

        എങ്കിൽ (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } അല്ലെങ്കിൽ {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        എങ്കിൽ (ibv_post_send(c->qp, &wr, &bad) != 0) തിരികെ അസത്യം;
        c->last_active = std::chrono::steady_clock::now();
        ഘടന ibv_wc wc;
        എങ്കിൽ (!rdma_poll(c->scq, &wc)) തിരികെ അസത്യം;

        src += chunk;
        rem -= chunk;
    }
    തിരികെ സത്യം;
}

തർക്കം socket_t::impl::rdma_recv(ശൂന്യം * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    ആകുമ്പോൾ (rem > 0) {
        ഘടന ibv_wc wc;
        എങ്കിൽ (!rdma_poll(c->rcq, &wc)) തിരികെ അസത്യം;

        പൂർണ്ണസംഖ്യ slot = (പൂർണ്ണസംഖ്യ)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        എങ്കിൽ (!c->post_rx(slot)) തിരികെ അസത്യം;

        dst += got;
        rem -= got;
    }
    തിരികെ സത്യം;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

തർക്കം socket_t::impl::send_data(സ്ഥിരം ശൂന്യം * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    എങ്കിൽ (use_rdma) {
        തിരികെ rdma->send(data, size);
    }
#അല്ലെങ്കിലെങ്കിൽ defined(GGML_RPC_RDMA)
    എങ്കിൽ (use_rdma) {
        തിരികെ rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    ആകുമ്പോൾ (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (സ്ഥിരം അക്ഷരം *)data + bytes_sent, size_to_send, 0);
        എങ്കിൽ (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            തിരികെ അസത്യം;
        }
        bytes_sent += (size_t)n;
    }
    തിരികെ സത്യം;
}

തർക്കം socket_t::impl::recv_data(ശൂന്യം * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    എങ്കിൽ (use_rdma) {
        തിരികെ rdma->recv(data, size);
    }
#അല്ലെങ്കിലെങ്കിൽ defined(GGML_RPC_RDMA)
    എങ്കിൽ (use_rdma) {
        തിരികെ rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    ആകുമ്പോൾ (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (അക്ഷരം *)data + bytes_recv, size_to_recv, 0);
        എങ്കിൽ (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            തിരികെ അസത്യം;
        }
        എങ്കിൽ (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            തിരികെ അസത്യം;
        }
        bytes_recv += (size_t)n;
    }
    തിരികെ സത്യം;
}

ശൂന്യം socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    എങ്കിൽ (std::getenv("GGML_RPC_NO_RDMA")) {
        തിരികെ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    എങ്കിൽ (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    എങ്കിൽ (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  അല്ലെങ്കിൽ
    rdma_local = {};
    എങ്കിൽ (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, വലുപ്പം(rc));
    } അല്ലെങ്കിൽ {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

ശൂന്യം socket_t::impl::update_caps(സ്ഥിരം uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    തർക്കം remote_rdma = അസത്യം;
    വേണ്ടി (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    എങ്കിൽ (!rdma || !remote_rdma) {
        എങ്കിൽ (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        തിരികെ;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    തർക്കം activated = rdma->activate(remote_caps);
#  അല്ലെങ്കിൽ
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, വലുപ്പം(rc));
    തർക്കം activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    എങ്കിൽ (activated) {
        use_rdma = സത്യം;
    } അല്ലെങ്കിൽ {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#അല്ലെങ്കിൽ
    (ശൂന്യം)remote_caps;
#endif // GGML_RPC_RDMA
}

തർക്കം socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    എങ്കിൽ (use_rdma) {
        തിരികെ rdma->flush();
    }
#endif
    തിരികെ സത്യം;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = സ്ഥിരസ്ഥിതി;

തർക്കം socket_t::send_data(സ്ഥിരം ശൂന്യം * data, size_t size) {
    തിരികെ pimpl->send_data(data, size);
}

തർക്കം socket_t::recv_data(ശൂന്യം * data, size_t size) {
    തിരികെ pimpl->recv_data(data, size);
}

തർക്കം socket_t::flush() {
    തിരികെ pimpl->flush();
}

ശൂന്യം socket_t::get_caps(uint8_t * local_caps) {
    തിരികെ pimpl->get_caps(local_caps);
}

ശൂന്യം socket_t::update_caps(സ്ഥിരം uint8_t * remote_caps) {
    തിരികെ pimpl->update_caps(remote_caps);
}

സ്ഥിരം അക്ഷരം * socket_t::transport_name() സ്ഥിരം {
    തിരികെ pimpl->use_rdma ? "RDMA" : "TCP";
}

സ്ഥിതം തർക്കം is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    തിരികെ sockfd != INVALID_SOCKET;
#അല്ലെങ്കിൽ
    തിരികെ sockfd >= 0;
#endif
}

സ്ഥിതം തർക്കം set_no_delay(sockfd_t sockfd) {
    പൂർണ്ണസംഖ്യ flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    പൂർണ്ണസംഖ്യ ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (അക്ഷരം *)&flag, വലുപ്പം(പൂർണ്ണസംഖ്യ));
    തിരികെ ret == 0;
}

സ്ഥിതം തർക്കം set_reuse_addr(sockfd_t sockfd) {
    പൂർണ്ണസംഖ്യ flag = 1;
    പൂർണ്ണസംഖ്യ ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (അക്ഷരം *)&flag, വലുപ്പം(പൂർണ്ണസംഖ്യ));
    തിരികെ ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, ശൂന്യ, ശൂന്യ);
    എങ്കിൽ (!is_valid_fd(client_socket_fd)) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        തിരികെ nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    തിരികെ socket_ptr(പുതിയ socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(സ്ഥിരം അക്ഷരം * host, പൂർണ്ണസംഖ്യ port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    എങ്കിൽ (!is_valid_fd(sockfd)) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        തിരികെ nullptr;
    }
    എങ്കിൽ (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        തിരികെ nullptr;
    }
    ഘടന sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    എങ്കിൽ (bind(sockfd, (ഘടന sockaddr *) &serv_addr, വലുപ്പം(serv_addr)) < 0) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (listen(sockfd, 1) < 0) {
        തിരികെ nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    തിരികെ socket_ptr(പുതിയ socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(സ്ഥിരം അക്ഷരം * host, പൂർണ്ണസംഖ്യ port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    എങ്കിൽ (!is_valid_fd(sockfd)) {
        തിരികെ nullptr;
    }
    എങ്കിൽ (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        തിരികെ nullptr;
    }
    ഘടന sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ഘടന hostent * server = gethostbyname(host);
    എങ്കിൽ (server == ശൂന്യ) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        തിരികെ nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    എങ്കിൽ (::connect(sockfd, (ഘടന sockaddr *)&addr, വലുപ്പം(addr)) < 0) {
        തിരികെ nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    തിരികെ socket_ptr(പുതിയ socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
സ്ഥിതം std::mutex g_rpc_transport_mu;
സ്ഥിതം തർക്കം g_rpc_transport_wsa_started = അസത്യം;
#endif

തർക്കം rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    എങ്കിൽ (g_rpc_transport_wsa_started) {
        തിരികെ സത്യം;
    }
    WSADATA wsaData;
    പൂർണ്ണസംഖ്യ res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    എങ്കിൽ (res != 0) {
        തിരികെ അസത്യം;
    }
    g_rpc_transport_wsa_started = സത്യം;
    തിരികെ സത്യം;
#അല്ലെങ്കിൽ
    തിരികെ സത്യം;
#endif
}

ശൂന്യം rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    എങ്കിൽ (!g_rpc_transport_wsa_started) {
        തിരികെ;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = അസത്യം;
#endif
}

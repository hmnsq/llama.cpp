#અન્તર્ભાવ "transport.h"
#અન્તર્ભાવ "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#નહીંતર
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <optional>

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
પ્રકારવ્યાખ્યા SOCKET sockfd_t;
using ssize_t = __int64;
#નહીંતર
પ્રકારવ્યાખ્યા પૂર્ણાંક sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
સ્થિર constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#જો defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
સ્થિર constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
સ્થિર constexpr પૂર્ણાંક    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
સ્થિર constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

સંરચના rdma_conn {
    સંરચના ibv_context * ctx = nullptr;
    સંરચના ibv_pd * pd  = nullptr;
    સંરચના ibv_cq * scq = nullptr;   // send completions
    સંરચના ibv_cq * rcq = nullptr;   // recv completions
    સંરચના ibv_qp * qp  = nullptr;
    સંરચના ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    શૂન્ય          * tx_buf = nullptr;
    સંરચના ibv_mr * tx_mr  = nullptr;

    શૂન્ય          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    સંરચના ibv_mr * rx_mr  = nullptr;
    પૂર્ણાંક             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(પૂર્ણાંક i) અચલ {
        પાછા static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    તાર્કિક post_rx(પૂર્ણાંક i) {
        સંરચના ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        સંરચના ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.અંક_sge = 1;
        પાછા ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        જો (tx_mr) ibv_dereg_mr(tx_mr);
        જો (rx_mr) ibv_dereg_mr(rx_mr);
        મુક્ત(tx_buf);
        મુક્ત(rx_buf);
        જો (qp)  ibv_destroy_qp(qp);
        જો (scq) ibv_destroy_cq(scq);
        જો (rcq) ibv_destroy_cq(rcq);
        જો (ch)  ibv_destroy_comp_channel(ch);
        જો (pd)  ibv_dealloc_pd(pd);
        જો (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
સંરચના rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    પૂર્ણાંક      gid_idx = 0;
    ગણના ibv_mtu path_mtu = IBV_MTU_1024;
};

સંરચના rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(કદ(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

સંરચના socket_t::impl {
    impl(sockfd_t fd) : use_rdma(ખોટું), fd(fd) {}
    ~impl();
    તાર્કિક send_data(અચલ શૂન્ય * data, size_t size);
    તાર્કિક recv_data(શૂન્ય * data, size_t size);
    તાર્કિક flush();
    શૂન્ય get_caps(uint8_t * local_caps);
    શૂન્ય update_caps(અચલ uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  નહીંતર
    તાર્કિક rdma_probe();
    તાર્કિક rdma_send(અચલ શૂન્ય * data, size_t size);
    તાર્કિક rdma_recv(શૂન્ય * data, size_t size);
    તાર્કિક tcp_peer_closed();
    તાર્કિક rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, અચલ uint8_t * remote_gid);
    તાર્કિક rdma_poll(સંરચના ibv_cq * cq, સંરચના ibv_wc * wc);
    તાર્કિક rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    તાર્કિક     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, આ->fd);
#ifdef _WIN32
    જો (fd != INVALID_SOCKET) closesocket(આ->fd);
#નહીંતર
    જો (fd >= 0) close(આ->fd);
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
    socklen_t addr_len = કદ(addr);
    જો (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        પાછા std::nullopt;
    }
    rdma_gid_t target = {};
    જો (addr.ss_family == AF_INET) {
        અચલ auto * a = reinterpret_cast<અચલ sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        પાછા target;
    }
    જો (addr.ss_family == AF_INET6) {
        અચલ auto * a = reinterpret_cast<અચલ sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        પાછા target;
    }
    પાછા std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

તાર્કિક socket_t::impl::tcp_peer_closed() {
    જો (fd < 0) પાછા ખોટું;
#ifndef _WIN32
    સંરચના pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    પૂર્ણાંક r = poll(&pfd, 1, 0);
    પાછા r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#નહીંતર
    પાછા ખોટું;
#endif
}

તાર્કિક socket_t::impl::rdma_probe() {
    અચલ અક્ષર * dev_env = std::getenv("GGML_RDMA_DEV");
    અચલ અક્ષર * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    જો (!target_gid) {
        પાછા ખોટું;
    }

    અચલ uint8_t ib_port = 1;
    પૂર્ણાંક અંક_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&અંક_devs);
    જો (!devs || અંક_devs == 0) પાછા ખોટું;

    ibv_context * ibctx = nullptr;
    અચલ અક્ષર * matched_dev = nullptr;
    પૂર્ણાંક gid_idx = gid_env ? atoi(gid_env) : -1;
    પૂર્ણાંક gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    માટે (પૂર્ણાંક d = 0; d < અંક_devs; d++) {
        અચલ અક્ષર * dn = ibv_get_device_name(devs[d]);
        જો (dev_env && strcmp(dev_env, dn) != 0) ચાલુ;

        ibv_context * ctx = ibv_open_device(devs[d]);
        જો (!ctx) ચાલુ;

        ibv_port_attr pa;
        જો (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); ચાલુ; }

        પૂર્ણાંક found_gid = gid_idx;
        પૂર્ણાંક found_version = IBV_GID_TYPE_IB;
        જો (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            પૂર્ણાંક v2_idx = -1;
            પૂર્ણાંક v1_idx = -1;
            માટે (પૂર્ણાંક i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                જો (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) ચાલુ;
                જો (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) ચાલુ;
                જો (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } નહીંતર જો (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            જો (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } નહીંતર જો (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } નહીંતર {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            જો (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        જો (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            તોડો;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    જો (!ibctx) પાછા ખોટું;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    જો (!rdma->pd) પાછા ખોટું;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    જો (!rdma->scq || !rdma->rcq) પાછા ખોટું;

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
    જો (!rdma->qp) પાછા ખોટું;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    જો (!rdma->tx_buf || !rdma->rx_buf) પાછા ખોટું;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    જો (!rdma->tx_mr || !rdma->rx_mr) પાછા ખોટું;

    ibv_gid local_gid;
    જો (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) પાછા ખોટું;

    rdma_local.qpn = rdma->qp->qp_અંક;
    rdma_local.psn = rdma->qp->qp_અંક & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    અચલ અક્ષર * ver_str = "";
    જો (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } નહીંતર જો (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    પાછા સાચું;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
તાર્કિક socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, અચલ uint8_t * remote_gid) {
    // RESET -> INIT
    {
        સંરચના ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_અંક        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        જો (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            પાછા ખોટું;
        }
    }

    માટે (પૂર્ણાંક i = 0; i < RDMA_RX_DEPTH; i++) {
        જો (!rdma->post_rx(i)) પાછા ખોટું;
    }

    // INIT -> RTR
    {
        સંરચના ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_અંક        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_અંક   = rdma_local.ib_port;
        જો (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            પાછા ખોટું;
        }
    }

    // RTR -> RTS
    {
        સંરચના ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        જો (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            પાછા ખોટું;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    પાછા સાચું;
}

// Sleep until the completion channel has an event or the TCP peer closes.
તાર્કિક socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    સંરચના pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    જો (poll(pfds, 2, -1) < 0) {
        પાછા errno == EINTR;
    }
    જો (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        પાછા ખોટું;
    }
    જો (pfds[0].revents & POLLIN) {
        સંરચના ibv_cq * ev_cq  = nullptr;
        શૂન્ય          * ev_ctx = nullptr;
        જો (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            પાછા ખોટું;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    પાછા સાચું;
}

તાર્કિક socket_t::impl::rdma_poll(સંરચના ibv_cq * cq, સંરચના ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    તાર્કિક armed = ખોટું;
    માટે (uint64_t s = 1; ; s++) {
        પૂર્ણાંક n = ibv_poll_cq(cq, 1, wc);
        જો (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            જો (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            પાછા wc->status == IBV_WC_SUCCESS;
        }
        જો (n < 0) પાછા ખોટું;
        જો (armed) {
            // armed and still empty: sleep until the next completion
            જો (!rdma_wait_event()) {
                પાછા ખોટું;
            }
            armed = ખોટું;
            ચાલુ;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        જો (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            જો (ibv_req_notify_cq(cq, 0) != 0) {
                પાછા ખોટું;
            }
            armed = સાચું;
            ચાલુ;
        }
        જો ((s & 0xFFFFF) == 0) {
            જો (tcp_peer_closed()) {
                પાછા ખોટું;
            }
        }
    }
}

તાર્કિક socket_t::impl::rdma_send(અચલ શૂન્ય * data, size_t size) {
    rdma_conn * c = rdma.get();
    અચલ uint8_t * src = (અચલ uint8_t *)data;
    size_t rem = size;
    જ્યારે (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        સંરચના ibv_sge sge = {};
        સંરચના ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.અંક_sge = 1;

        જો (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } નહીંતર {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        જો (ibv_post_send(c->qp, &wr, &bad) != 0) પાછા ખોટું;
        c->last_active = std::chrono::steady_clock::now();
        સંરચના ibv_wc wc;
        જો (!rdma_poll(c->scq, &wc)) પાછા ખોટું;

        src += chunk;
        rem -= chunk;
    }
    પાછા સાચું;
}

તાર્કિક socket_t::impl::rdma_recv(શૂન્ય * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    જ્યારે (rem > 0) {
        સંરચના ibv_wc wc;
        જો (!rdma_poll(c->rcq, &wc)) પાછા ખોટું;

        પૂર્ણાંક slot = (પૂર્ણાંક)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        જો (!c->post_rx(slot)) પાછા ખોટું;

        dst += got;
        rem -= got;
    }
    પાછા સાચું;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

તાર્કિક socket_t::impl::send_data(અચલ શૂન્ય * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    જો (use_rdma) {
        પાછા rdma->send(data, size);
    }
#અથવાજો defined(GGML_RPC_RDMA)
    જો (use_rdma) {
        પાછા rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    જ્યારે (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (અચલ અક્ષર *)data + bytes_sent, size_to_send, 0);
        જો (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            પાછા ખોટું;
        }
        bytes_sent += (size_t)n;
    }
    પાછા સાચું;
}

તાર્કિક socket_t::impl::recv_data(શૂન્ય * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    જો (use_rdma) {
        પાછા rdma->recv(data, size);
    }
#અથવાજો defined(GGML_RPC_RDMA)
    જો (use_rdma) {
        પાછા rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    જ્યારે (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (અક્ષર *)data + bytes_recv, size_to_recv, 0);
        જો (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            પાછા ખોટું;
        }
        જો (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            પાછા ખોટું;
        }
        bytes_recv += (size_t)n;
    }
    પાછા સાચું;
}

શૂન્ય socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    જો (std::getenv("GGML_RPC_NO_RDMA")) {
        પાછા;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    જો (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    જો (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  નહીંતર
    rdma_local = {};
    જો (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, કદ(rc));
    } નહીંતર {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

શૂન્ય socket_t::impl::update_caps(અચલ uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    તાર્કિક remote_rdma = ખોટું;
    માટે (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    જો (!rdma || !remote_rdma) {
        જો (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        પાછા;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    તાર્કિક activated = rdma->activate(remote_caps);
#  નહીંતર
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, કદ(rc));
    તાર્કિક activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    જો (activated) {
        use_rdma = સાચું;
    } નહીંતર {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#નહીંતર
    (શૂન્ય)remote_caps;
#endif // GGML_RPC_RDMA
}

તાર્કિક socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    જો (use_rdma) {
        પાછા rdma->flush();
    }
#endif
    પાછા સાચું;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = મૂળભૂત;

તાર્કિક socket_t::send_data(અચલ શૂન્ય * data, size_t size) {
    પાછા pimpl->send_data(data, size);
}

તાર્કિક socket_t::recv_data(શૂન્ય * data, size_t size) {
    પાછા pimpl->recv_data(data, size);
}

તાર્કિક socket_t::flush() {
    પાછા pimpl->flush();
}

શૂન્ય socket_t::get_caps(uint8_t * local_caps) {
    પાછા pimpl->get_caps(local_caps);
}

શૂન્ય socket_t::update_caps(અચલ uint8_t * remote_caps) {
    પાછા pimpl->update_caps(remote_caps);
}

અચલ અક્ષર * socket_t::transport_name() અચલ {
    પાછા pimpl->use_rdma ? "RDMA" : "TCP";
}

સ્થિર તાર્કિક is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    પાછા sockfd != INVALID_SOCKET;
#નહીંતર
    પાછા sockfd >= 0;
#endif
}

સ્થિર તાર્કિક set_no_delay(sockfd_t sockfd) {
    પૂર્ણાંક flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    પૂર્ણાંક ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (અક્ષર *)&flag, કદ(પૂર્ણાંક));
    પાછા ret == 0;
}

સ્થિર તાર્કિક set_reuse_addr(sockfd_t sockfd) {
    પૂર્ણાંક flag = 1;
    પૂર્ણાંક ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (અક્ષર *)&flag, કદ(પૂર્ણાંક));
    પાછા ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, ખાલી, ખાલી);
    જો (!is_valid_fd(client_socket_fd)) {
        પાછા nullptr;
    }
    જો (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        પાછા nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    પાછા socket_ptr(નવું socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(અચલ અક્ષર * host, પૂર્ણાંક port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    જો (!is_valid_fd(sockfd)) {
        પાછા nullptr;
    }
    જો (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        પાછા nullptr;
    }
    જો (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        પાછા nullptr;
    }
    સંરચના sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    જો (bind(sockfd, (સંરચના sockaddr *) &serv_addr, કદ(serv_addr)) < 0) {
        પાછા nullptr;
    }
    જો (listen(sockfd, 1) < 0) {
        પાછા nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    પાછા socket_ptr(નવું socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(અચલ અક્ષર * host, પૂર્ણાંક port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    જો (!is_valid_fd(sockfd)) {
        પાછા nullptr;
    }
    જો (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        પાછા nullptr;
    }
    સંરચના sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    સંરચના hostent * server = gethostbyname(host);
    જો (server == ખાલી) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        પાછા nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    જો (::connect(sockfd, (સંરચના sockaddr *)&addr, કદ(addr)) < 0) {
        પાછા nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    પાછા socket_ptr(નવું socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
સ્થિર std::mutex g_rpc_transport_mu;
સ્થિર તાર્કિક g_rpc_transport_wsa_started = ખોટું;
#endif

તાર્કિક rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    જો (g_rpc_transport_wsa_started) {
        પાછા સાચું;
    }
    WSADATA wsaData;
    પૂર્ણાંક res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    જો (res != 0) {
        પાછા ખોટું;
    }
    g_rpc_transport_wsa_started = સાચું;
    પાછા સાચું;
#નહીંતર
    પાછા સાચું;
#endif
}

શૂન્ય rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    જો (!g_rpc_transport_wsa_started) {
        પાછા;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = ખોટું;
#endif
}

#அந்தர்பாவ "transport.h"
#அந்தர்பாவ "log.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#இல்லையெனில்
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <optional>

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
வகைவரை SOCKET sockfd_t;
using ssize_t = __int64;
#இல்லையெனில்
வகைவரை முழு sockfd_t;
#endif

#ifdef GGML_RPC_RDMA
நிலையான constexpr size_t RDMA_GID_SIZE = 16;            // RoCE GID / IB GID is always 16 bytes
using rdma_gid_t = std::array<uint8_t, RDMA_GID_SIZE>;
#endif // GGML_RPC_RDMA

#எனில் defined(GGML_RPC_RDMA) && !defined(GGML_RPC_RDMA_APPLE)
நிலையான constexpr size_t RDMA_CHUNK    = 256 * 1024;   // 256 KiB per send/recv (fits default 8 MiB memlock)
நிலையான constexpr முழு    RDMA_RX_DEPTH = 24;            // pre-posted recv ring: 24 × 256 KiB = 6 MiB
// keep polling the CQ for this long after the last activity, then sleep until the next completion
நிலையான constexpr auto   RDMA_SPIN_TIME = std::chrono::milliseconds(100);

கட்டமைப்பு rdma_conn {
    கட்டமைப்பு ibv_context * ctx = nullptr;
    கட்டமைப்பு ibv_pd * pd  = nullptr;
    கட்டமைப்பு ibv_cq * scq = nullptr;   // send completions
    கட்டமைப்பு ibv_cq * rcq = nullptr;   // recv completions
    கட்டமைப்பு ibv_qp * qp  = nullptr;
    கட்டமைப்பு ibv_comp_channel * ch = nullptr; // CQ events, so an idle connection can sleep instead of spinning

    std::chrono::steady_clock::time_point last_active; // last completion or posted send

    வெறுமை          * tx_buf = nullptr;
    கட்டமைப்பு ibv_mr * tx_mr  = nullptr;

    வெறுமை          * rx_buf = nullptr; // RDMA_RX_DEPTH × RDMA_CHUNK contiguous
    கட்டமைப்பு ibv_mr * rx_mr  = nullptr;
    முழு             rx_head = 0;

    uint32_t        max_inline = 0;

    uint8_t * rx_slot(முழு i) மாறிலி {
        திரும்பு static_cast<uint8_t *>(rx_buf) + static_cast<size_t>(i) * RDMA_CHUNK;
    }

    தருக்க post_rx(முழு i) {
        கட்டமைப்பு ibv_sge sge = {};
        sge.addr   = (uintptr_t)rx_slot(i);
        sge.length = RDMA_CHUNK;
        sge.lkey   = rx_mr->lkey;
        கட்டமைப்பு ibv_recv_wr wr = {}, * bad = nullptr;
        wr.wr_id   = (uint64_t)i;
        wr.sg_list = &sge;
        wr.இலக்கம்_sge = 1;
        திரும்பு ibv_post_recv(qp, &wr, &bad) == 0;
    }

    ~rdma_conn() {
        எனில் (tx_mr) ibv_dereg_mr(tx_mr);
        எனில் (rx_mr) ibv_dereg_mr(rx_mr);
        விடுவி(tx_buf);
        விடுவி(rx_buf);
        எனில் (qp)  ibv_destroy_qp(qp);
        எனில் (scq) ibv_destroy_cq(scq);
        எனில் (rcq) ibv_destroy_cq(rcq);
        எனில் (ch)  ibv_destroy_comp_channel(ch);
        எனில் (pd)  ibv_dealloc_pd(pd);
        எனில் (ctx) ibv_close_device(ctx);
    }
};

// Local RDMA parameters captured during the probe phase and later consumed
// by rdma_activate() after the remote side's caps arrive via HELLO.
கட்டமைப்பு rdma_local_info {
    uint32_t qpn     = 0;
    uint32_t psn     = 0;
    uint8_t  gid[RDMA_GID_SIZE] = {};
    uint8_t  ib_port = 0;
    முழு      gid_idx = 0;
    எண்ணிக்கை ibv_mtu path_mtu = IBV_MTU_1024;
};

கட்டமைப்பு rdma_caps {
    uint32_t qpn;
    uint32_t psn;
    uint8_t  gid[RDMA_GID_SIZE];
};

static_assert(அளவு(rdma_caps) == RPC_CONN_CAPS_SIZE, "rdma_caps must match conn_caps size");

#endif // GGML_RPC_RDMA && !GGML_RPC_RDMA_APPLE

கட்டமைப்பு socket_t::impl {
    impl(sockfd_t fd) : use_rdma(பொய்), fd(fd) {}
    ~impl();
    தருக்க send_data(மாறிலி வெறுமை * data, size_t size);
    தருக்க recv_data(வெறுமை * data, size_t size);
    தருக்க flush();
    வெறுமை get_caps(uint8_t * local_caps);
    வெறுமை update_caps(மாறிலி uint8_t * remote_caps);

#ifdef GGML_RPC_RDMA
    std::optional<rdma_gid_t> rdma_build_target_gid();

#  ifdef GGML_RPC_RDMA_APPLE
    std::unique_ptr<apple_rdma> rdma;
#  இல்லையெனில்
    தருக்க rdma_probe();
    தருக்க rdma_send(மாறிலி வெறுமை * data, size_t size);
    தருக்க rdma_recv(வெறுமை * data, size_t size);
    தருக்க tcp_peer_closed();
    தருக்க rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, மாறிலி uint8_t * remote_gid);
    தருக்க rdma_poll(கட்டமைப்பு ibv_cq * cq, கட்டமைப்பு ibv_wc * wc);
    தருக்க rdma_wait_event();

    std::unique_ptr<rdma_conn> rdma;
    rdma_local_info            rdma_local = {};
#  endif
#endif // GGML_RPC_RDMA
    தருக்க     use_rdma;
    sockfd_t fd;
};

socket_t::impl::~impl() {
#ifdef GGML_RPC_RDMA
    rdma.reset();
#endif // GGML_RPC_RDMA
    LOG_DBG("[%s] closing socket %d\n", __func__, இது->fd);
#ifdef _WIN32
    எனில் (fd != INVALID_SOCKET) closesocket(இது->fd);
#இல்லையெனில்
    எனில் (fd >= 0) close(இது->fd);
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
    socklen_t addr_len = அளவு(addr);
    எனில் (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) != 0) {
        திரும்பு std::nullopt;
    }
    rdma_gid_t target = {};
    எனில் (addr.ss_family == AF_INET) {
        மாறிலி auto * a = reinterpret_cast<மாறிலி sockaddr_in *>(&addr);
        target[10] = 0xff;
        target[11] = 0xff;
        memcpy(&target[12], &a->sin_addr, 4);
        திரும்பு target;
    }
    எனில் (addr.ss_family == AF_INET6) {
        மாறிலி auto * a = reinterpret_cast<மாறிலி sockaddr_in6 *>(&addr);
        memcpy(target.data(), &a->sin6_addr, RDMA_GID_SIZE);
        திரும்பு target;
    }
    திரும்பு std::nullopt;
}

#ifndef GGML_RPC_RDMA_APPLE

தருக்க socket_t::impl::tcp_peer_closed() {
    எனில் (fd < 0) திரும்பு பொய்;
#ifndef _WIN32
    கட்டமைப்பு pollfd pfd = { fd, POLLIN | POLLRDHUP, 0 };
    முழு r = poll(&pfd, 1, 0);
    திரும்பு r > 0 && (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP));
#இல்லையெனில்
    திரும்பு பொய்;
#endif
}

தருக்க socket_t::impl::rdma_probe() {
    மாறிலி எழுத்து * dev_env = std::getenv("GGML_RDMA_DEV");
    மாறிலி எழுத்து * gid_env = std::getenv("GGML_RDMA_GID");

    auto target_gid = rdma_build_target_gid();
    எனில் (!target_gid) {
        திரும்பு பொய்;
    }

    மாறிலி uint8_t ib_port = 1;
    முழு இலக்கம்_devs = 0;
    ibv_device ** devs = ibv_get_device_list(&இலக்கம்_devs);
    எனில் (!devs || இலக்கம்_devs == 0) திரும்பு பொய்;

    ibv_context * ibctx = nullptr;
    மாறிலி எழுத்து * matched_dev = nullptr;
    முழு gid_idx = gid_env ? atoi(gid_env) : -1;
    முழு gid_version = IBV_GID_TYPE_IB;  // 0 = unknown/IB

    க்கு (முழு d = 0; d < இலக்கம்_devs; d++) {
        மாறிலி எழுத்து * dn = ibv_get_device_name(devs[d]);
        எனில் (dev_env && strcmp(dev_env, dn) != 0) தொடர்;

        ibv_context * ctx = ibv_open_device(devs[d]);
        எனில் (!ctx) தொடர்;

        ibv_port_attr pa;
        எனில் (ibv_query_port(ctx, ib_port, &pa) != 0) { ibv_close_device(ctx); தொடர்; }

        முழு found_gid = gid_idx;
        முழு found_version = IBV_GID_TYPE_IB;
        எனில் (found_gid < 0) {
            // Find a GID on this port whose bytes equal the local TCP address
            // (IPv4 or IPv6). Prefer RoCE v2 (UDP/IP, L3-routable) over v1
            // (raw Ethernet, same-L2 only) so silent hangs on L3-routed paths
            // are avoided. ibv_query_gid_ex returns gid+type in one call.
            முழு v2_idx = -1;
            முழு v1_idx = -1;
            க்கு (முழு i = 0; i < pa.gid_tbl_len; i++) {
                ibv_gid_entry entry = {};
                எனில் (ibv_query_gid_ex(ctx, ib_port, i, &entry, 0) != 0) தொடர்;
                எனில் (memcmp(entry.gid.raw, target_gid->data(), RDMA_GID_SIZE) != 0) தொடர்;
                எனில் (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && v2_idx < 0) {
                    v2_idx = i;
                } இல்லையெனில் எனில் (entry.gid_type == IBV_GID_TYPE_ROCE_V1 && v1_idx < 0) {
                    v1_idx = i;
                }
            }
            எனில் (v2_idx >= 0) {
                found_gid = v2_idx;
                found_version = IBV_GID_TYPE_ROCE_V2;
            } இல்லையெனில் எனில் (v1_idx >= 0) {
                found_gid = v1_idx;
                found_version = IBV_GID_TYPE_ROCE_V1;
            }
        } இல்லையெனில் {
            // Explicit GID index from GGML_RDMA_GID — fetch its type for logging.
            ibv_gid_entry entry = {};
            எனில் (ibv_query_gid_ex(ctx, ib_port, found_gid, &entry, 0) == 0) {
                found_version = entry.gid_type;
            }
        }
        எனில் (found_gid >= 0) {
            ibctx = ctx;
            gid_idx = found_gid;
            gid_version = found_version;
            matched_dev = dn;
            rdma_local.path_mtu = pa.active_mtu;
            நிறுத்து;
        }
        ibv_close_device(ctx);
    }
    ibv_free_device_list(devs);
    எனில் (!ibctx) திரும்பு பொய்;

    rdma_local.ib_port = ib_port;
    rdma_local.gid_idx = gid_idx;

    rdma = std::make_unique<rdma_conn>();
    rdma->ctx = ibctx;

    rdma->pd = ibv_alloc_pd(ibctx);
    எனில் (!rdma->pd) திரும்பு பொய்;

    // without a completion channel rdma_poll() spins all the time, as before
    rdma->ch  = ibv_create_comp_channel(ibctx);
    rdma->scq = ibv_create_cq(ibctx, 16, nullptr, rdma->ch, 0);
    rdma->rcq = ibv_create_cq(ibctx, RDMA_RX_DEPTH + 4, nullptr, rdma->ch, 0);
    எனில் (!rdma->scq || !rdma->rcq) திரும்பு பொய்;

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
    எனில் (!rdma->qp) திரும்பு பொய்;
    rdma->max_inline = qia.cap.max_inline_data;

    rdma->tx_buf = aligned_alloc(4096, RDMA_CHUNK);
    rdma->rx_buf = aligned_alloc(4096, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK);
    எனில் (!rdma->tx_buf || !rdma->rx_buf) திரும்பு பொய்;

    rdma->tx_mr = ibv_reg_mr(rdma->pd, rdma->tx_buf, RDMA_CHUNK, IBV_ACCESS_LOCAL_WRITE);
    rdma->rx_mr = ibv_reg_mr(rdma->pd, rdma->rx_buf, static_cast<size_t>(RDMA_RX_DEPTH) * RDMA_CHUNK,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    எனில் (!rdma->tx_mr || !rdma->rx_mr) திரும்பு பொய்;

    ibv_gid local_gid;
    எனில் (ibv_query_gid(ibctx, ib_port, gid_idx, &local_gid) != 0) திரும்பு பொய்;

    rdma_local.qpn = rdma->qp->qp_இலக்கம்;
    rdma_local.psn = rdma->qp->qp_இலக்கம் & 0xffffff;
    memcpy(&rdma_local.gid, &local_gid, RDMA_GID_SIZE);

    மாறிலி எழுத்து * ver_str = "";
    எனில் (gid_version == IBV_GID_TYPE_ROCE_V2) {
        ver_str = " RoCEv2";
    } இல்லையெனில் எனில் (gid_version == IBV_GID_TYPE_ROCE_V1) {
        ver_str = " RoCEv1";
    }
    LOG_INFO("RDMA probed: dev=%s gid=%d%s qpn=%u inline=%u\n",
                  matched_dev, gid_idx, ver_str, rdma_local.qpn, rdma->max_inline);
    திரும்பு உண்மை;
}

// Phase 2: Given remote QPN/PSN/GID, transition QP: RESET->INIT->pre-post->RTR->RTS.
// On success, the connection is live and ready for rdma_send/rdma_recv.
தருக்க socket_t::impl::rdma_activate(uint32_t remote_qpn, uint32_t remote_psn, மாறிலி uint8_t * remote_gid) {
    // RESET -> INIT
    {
        கட்டமைப்பு ibv_qp_attr a = {};
        a.qp_state        = IBV_QPS_INIT;
        a.port_இலக்கம்        = rdma_local.ib_port;
        a.pkey_index      = 0;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        எனில் (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            திரும்பு பொய்;
        }
    }

    க்கு (முழு i = 0; i < RDMA_RX_DEPTH; i++) {
        எனில் (!rdma->post_rx(i)) திரும்பு பொய்;
    }

    // INIT -> RTR
    {
        கட்டமைப்பு ibv_qp_attr a = {};
        a.qp_state           = IBV_QPS_RTR;
        a.path_mtu           = rdma_local.path_mtu;
        a.dest_qp_இலக்கம்        = remote_qpn;
        a.rq_psn             = remote_psn;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer      = 1;
        a.ah_attr.is_global  = 1;
        memcpy(&a.ah_attr.grh.dgid, remote_gid, RDMA_GID_SIZE);
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.sgid_index = rdma_local.gid_idx;
        a.ah_attr.dlid       = 0;
        a.ah_attr.port_இலக்கம்   = rdma_local.ib_port;
        எனில் (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            திரும்பு பொய்;
        }
    }

    // RTR -> RTS
    {
        கட்டமைப்பு ibv_qp_attr a = {};
        a.qp_state     = IBV_QPS_RTS;
        a.timeout      = 14;
        a.retry_cnt    = 7;
        a.rnr_retry    = 7;
        a.sq_psn       = rdma_local.psn;
        a.max_rd_atomic = 1;
        எனில் (ibv_modify_qp(rdma->qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            திரும்பு பொய்;
        }
    }

    rdma->last_active = std::chrono::steady_clock::now();

    LOG_INFO("RDMA activated: qpn=%u->%u mtu=%d rx_depth=%d\n",
                  rdma_local.qpn, remote_qpn, 128 << rdma_local.path_mtu, RDMA_RX_DEPTH);
    திரும்பு உண்மை;
}

// Sleep until the completion channel has an event or the TCP peer closes.
தருக்க socket_t::impl::rdma_wait_event() {
    rdma_conn * c = rdma.get();
    // POLLHUP and POLLERR are always reported, the TCP socket carries no data after the RDMA upgrade
    கட்டமைப்பு pollfd pfds[2] = {
        { c->ch->fd, POLLIN,    0 },
        { fd,        POLLRDHUP, 0 },
    };
    எனில் (poll(pfds, 2, -1) < 0) {
        திரும்பு errno == EINTR;
    }
    எனில் (pfds[1].revents & (POLLHUP | POLLERR | POLLRDHUP)) {
        திரும்பு பொய்;
    }
    எனில் (pfds[0].revents & POLLIN) {
        கட்டமைப்பு ibv_cq * ev_cq  = nullptr;
        வெறுமை          * ev_ctx = nullptr;
        எனில் (ibv_get_cq_event(c->ch, &ev_cq, &ev_ctx) != 0) {
            திரும்பு பொய்;
        }
        ibv_ack_cq_events(ev_cq, 1);
    }
    திரும்பு உண்மை;
}

தருக்க socket_t::impl::rdma_poll(கட்டமைப்பு ibv_cq * cq, கட்டமைப்பு ibv_wc * wc) {
    rdma_conn * c = rdma.get();
    தருக்க armed = பொய்;
    க்கு (uint64_t s = 1; ; s++) {
        முழு n = ibv_poll_cq(cq, 1, wc);
        எனில் (n > 0) {
            c->last_active = std::chrono::steady_clock::now();
            எனில் (wc->status != IBV_WC_SUCCESS) {
                LOG_ERROR("RDMA CQ wc error: status=%d (%s) vendor_err=0x%x\n",
                    wc->status, ibv_wc_status_str(wc->status), wc->vendor_err);
            }
            திரும்பு wc->status == IBV_WC_SUCCESS;
        }
        எனில் (n < 0) திரும்பு பொய்;
        எனில் (armed) {
            // armed and still empty: sleep until the next completion
            எனில் (!rdma_wait_event()) {
                திரும்பு பொய்;
            }
            armed = பொய்;
            தொடர்;
        }
        // spin while the connection is busy, arm the CQ once it has been idle for RDMA_SPIN_TIME
        // a completion that arrives before arming raises no event, so poll once more after arming
        எனில் (c->ch && (s & 0x3FF) == 0 && std::chrono::steady_clock::now() - c->last_active > RDMA_SPIN_TIME) {
            எனில் (ibv_req_notify_cq(cq, 0) != 0) {
                திரும்பு பொய்;
            }
            armed = உண்மை;
            தொடர்;
        }
        எனில் ((s & 0xFFFFF) == 0) {
            எனில் (tcp_peer_closed()) {
                திரும்பு பொய்;
            }
        }
    }
}

தருக்க socket_t::impl::rdma_send(மாறிலி வெறுமை * data, size_t size) {
    rdma_conn * c = rdma.get();
    மாறிலி uint8_t * src = (மாறிலி uint8_t *)data;
    size_t rem = size;
    போது (rem > 0) {
        size_t chunk = std::min(rem, RDMA_CHUNK);

        கட்டமைப்பு ibv_sge sge = {};
        கட்டமைப்பு ibv_send_wr wr = {}, * bad = nullptr;
        wr.opcode  = IBV_WR_SEND;
        wr.sg_list = &sge;
        wr.இலக்கம்_sge = 1;

        எனில் (chunk <= c->max_inline) {
            sge.addr   = (uintptr_t)src;
            sge.length = chunk;
            wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        } இல்லையெனில் {
            memcpy(c->tx_buf, src, chunk);
            sge.addr   = (uintptr_t)c->tx_buf;
            sge.length = chunk;
            sge.lkey   = c->tx_mr->lkey;
            wr.send_flags = IBV_SEND_SIGNALED;
        }

        எனில் (ibv_post_send(c->qp, &wr, &bad) != 0) திரும்பு பொய்;
        c->last_active = std::chrono::steady_clock::now();
        கட்டமைப்பு ibv_wc wc;
        எனில் (!rdma_poll(c->scq, &wc)) திரும்பு பொய்;

        src += chunk;
        rem -= chunk;
    }
    திரும்பு உண்மை;
}

தருக்க socket_t::impl::rdma_recv(வெறுமை * data, size_t size) {
    rdma_conn * c = rdma.get();
    uint8_t * dst = (uint8_t *)data;
    size_t rem = size;
    போது (rem > 0) {
        கட்டமைப்பு ibv_wc wc;
        எனில் (!rdma_poll(c->rcq, &wc)) திரும்பு பொய்;

        முழு slot = (முழு)wc.wr_id;
        size_t got = wc.byte_len;
        memcpy(dst, c->rx_slot(slot), got);

        எனில் (!c->post_rx(slot)) திரும்பு பொய்;

        dst += got;
        rem -= got;
    }
    திரும்பு உண்மை;
}

#endif // !GGML_RPC_RDMA_APPLE (Linux RC transport)

#endif // GGML_RPC_RDMA

தருக்க socket_t::impl::send_data(மாறிலி வெறுமை * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    எனில் (use_rdma) {
        திரும்பு rdma->send(data, size);
    }
#அல்லதெனில் defined(GGML_RPC_RDMA)
    எனில் (use_rdma) {
        திரும்பு rdma_send(data, size);
    }
#endif
    size_t bytes_sent = 0;
    போது (bytes_sent < size) {
        size_t size_to_send = std::min(size - bytes_sent, MAX_CHUNK_SIZE);
        ssize_t n = send(fd, (மாறிலி எழுத்து *)data + bytes_sent, size_to_send, 0);
        எனில் (n < 0) {
            LOG_ERROR("send failed (bytes_sent=%zu, size_to_send=%zu)\n",
                           bytes_sent, size_to_send);
            திரும்பு பொய்;
        }
        bytes_sent += (size_t)n;
    }
    திரும்பு உண்மை;
}

தருக்க socket_t::impl::recv_data(வெறுமை * data, size_t size) {
    LOG_DBG3("[%s] transport: %s, size: %s\n", __func__, use_rdma ? "RDMA" : "TCP", rpc_log_format_size(size).c_str());
#ifdef GGML_RPC_RDMA_APPLE
    எனில் (use_rdma) {
        திரும்பு rdma->recv(data, size);
    }
#அல்லதெனில் defined(GGML_RPC_RDMA)
    எனில் (use_rdma) {
        திரும்பு rdma_recv(data, size);
    }
#endif
    size_t bytes_recv = 0;
    போது (bytes_recv < size) {
        size_t size_to_recv = std::min(size - bytes_recv, MAX_CHUNK_SIZE);
        ssize_t n = recv(fd, (எழுத்து *)data + bytes_recv, size_to_recv, 0);
        எனில் (n < 0) {
            LOG_ERROR("recv failed (bytes_recv=%zu, size_to_recv=%zu)\n",
                           bytes_recv, size_to_recv);
            திரும்பு பொய்;
        }
        எனில் (n == 0) {
            LOG_WARN("recv returned 0 (peer closed?)\n");
            திரும்பு பொய்;
        }
        bytes_recv += (size_t)n;
    }
    திரும்பு உண்மை;
}

வெறுமை socket_t::impl::get_caps(uint8_t * local_caps) {
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
#ifdef GGML_RPC_RDMA
    எனில் (std::getenv("GGML_RPC_NO_RDMA")) {
        திரும்பு;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    auto target_gid = rdma_build_target_gid();
    எனில் (target_gid) {
        rdma = apple_rdma::probe(fd, target_gid->data(), local_caps);
    }
    எனில் (!rdma) {
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  இல்லையெனில்
    rdma_local = {};
    எனில் (rdma_probe()) {
        rdma_caps rc = {};
        rc.qpn = rdma_local.qpn;
        rc.psn = rdma_local.psn;
        memcpy(rc.gid, rdma_local.gid, RDMA_GID_SIZE);
        memcpy(local_caps, &rc, அளவு(rc));
    } இல்லையெனில் {
        rdma.reset();
        LOG_DBG2("[%s] RDMA probe failed, continuing with TCP\n", __func__);
    }
#  endif
#endif // GGML_RPC_RDMA
}

வெறுமை socket_t::impl::update_caps(மாறிலி uint8_t * remote_caps) {
#ifdef GGML_RPC_RDMA
    // a peer that has no RDMA advertises all-zero caps and takes no further part
    // in the negotiation, so drop to TCP without reporting a failure
    தருக்க remote_rdma = பொய்;
    க்கு (size_t i = 0; i < RPC_CONN_CAPS_SIZE; i++) {
        remote_rdma |= remote_caps[i] != 0;
    }
    எனில் (!rdma || !remote_rdma) {
        எனில் (rdma && !remote_rdma) {
            LOG_WARN("[%s] peer does not support RDMA, continuing with TCP\n", __func__);
        }
        rdma.reset();
        திரும்பு;
    }
#  ifdef GGML_RPC_RDMA_APPLE
    தருக்க activated = rdma->activate(remote_caps);
#  இல்லையெனில்
    rdma_caps rc = {};
    memcpy(&rc, remote_caps, அளவு(rc));
    தருக்க activated = rdma_activate(rc.qpn, rc.psn, rc.gid);
#  endif
    எனில் (activated) {
        use_rdma = உண்மை;
    } இல்லையெனில் {
        LOG_WARN("RDMA activate failed, staying on TCP\n");
        rdma.reset();
    }
#இல்லையெனில்
    (வெறுமை)remote_caps;
#endif // GGML_RPC_RDMA
}

தருக்க socket_t::impl::flush() {
#ifdef GGML_RPC_RDMA_APPLE
    எனில் (use_rdma) {
        திரும்பு rdma->flush();
    }
#endif
    திரும்பு உண்மை;
}

/////////////////////////////////////////////////////////////////////////////

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = இயல்பு;

தருக்க socket_t::send_data(மாறிலி வெறுமை * data, size_t size) {
    திரும்பு pimpl->send_data(data, size);
}

தருக்க socket_t::recv_data(வெறுமை * data, size_t size) {
    திரும்பு pimpl->recv_data(data, size);
}

தருக்க socket_t::flush() {
    திரும்பு pimpl->flush();
}

வெறுமை socket_t::get_caps(uint8_t * local_caps) {
    திரும்பு pimpl->get_caps(local_caps);
}

வெறுமை socket_t::update_caps(மாறிலி uint8_t * remote_caps) {
    திரும்பு pimpl->update_caps(remote_caps);
}

மாறிலி எழுத்து * socket_t::transport_name() மாறிலி {
    திரும்பு pimpl->use_rdma ? "RDMA" : "TCP";
}

நிலையான தருக்க is_valid_fd(sockfd_t sockfd) {
#ifdef _WIN32
    திரும்பு sockfd != INVALID_SOCKET;
#இல்லையெனில்
    திரும்பு sockfd >= 0;
#endif
}

நிலையான தருக்க set_no_delay(sockfd_t sockfd) {
    முழு flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    முழு ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (எழுத்து *)&flag, அளவு(முழு));
    திரும்பு ret == 0;
}

நிலையான தருக்க set_reuse_addr(sockfd_t sockfd) {
    முழு flag = 1;
    முழு ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (எழுத்து *)&flag, அளவு(முழு));
    திரும்பு ret == 0;
}

socket_ptr socket_t::accept() {
    auto client_socket_fd = ::accept(pimpl->fd, வெற்று, வெற்று);
    எனில் (!is_valid_fd(client_socket_fd)) {
        திரும்பு nullptr;
    }
    எனில் (!set_no_delay(client_socket_fd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        திரும்பு nullptr;
    }
    LOG_DBG("[%s] accepted client connection\n", __func__);
    திரும்பு socket_ptr(புதிய socket_t(std::make_unique<impl>(client_socket_fd)));
}

socket_ptr socket_t::create_server(மாறிலி எழுத்து * host, முழு port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    எனில் (!is_valid_fd(sockfd)) {
        திரும்பு nullptr;
    }
    எனில் (!set_reuse_addr(sockfd)) {
        LOG_ERROR("Failed to set SO_REUSEADDR\n");
        திரும்பு nullptr;
    }
    எனில் (inet_addr(host) == INADDR_NONE) {
        LOG_ERROR("Invalid host address: %s\n", host);
        திரும்பு nullptr;
    }
    கட்டமைப்பு sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    எனில் (bind(sockfd, (கட்டமைப்பு sockaddr *) &serv_addr, அளவு(serv_addr)) < 0) {
        திரும்பு nullptr;
    }
    எனில் (listen(sockfd, 1) < 0) {
        திரும்பு nullptr;
    }
    LOG_DBG("[%s] listening on %s:%d\n", __func__, host, port);
    திரும்பு socket_ptr(புதிய socket_t(std::make_unique<impl>(sockfd)));
}

socket_ptr socket_t::connect(மாறிலி எழுத்து * host, முழு port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    எனில் (!is_valid_fd(sockfd)) {
        திரும்பு nullptr;
    }
    எனில் (!set_no_delay(sockfd)) {
        LOG_ERROR("Failed to set TCP_NODELAY\n");
        திரும்பு nullptr;
    }
    கட்டமைப்பு sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    கட்டமைப்பு hostent * server = gethostbyname(host);
    எனில் (server == வெற்று) {
        LOG_ERROR("Cannot resolve host '%s'\n", host);
        திரும்பு nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    எனில் (::connect(sockfd, (கட்டமைப்பு sockaddr *)&addr, அளவு(addr)) < 0) {
        திரும்பு nullptr;
    }
    LOG_DBG("[%s] connected to %s:%d\n", __func__, host, port);
    திரும்பு socket_ptr(புதிய socket_t(std::make_unique<impl>(sockfd)));
}

#ifdef _WIN32
நிலையான std::mutex g_rpc_transport_mu;
நிலையான தருக்க g_rpc_transport_wsa_started = பொய்;
#endif

தருக்க rpc_transport_init() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    எனில் (g_rpc_transport_wsa_started) {
        திரும்பு உண்மை;
    }
    WSADATA wsaData;
    முழு res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    எனில் (res != 0) {
        திரும்பு பொய்;
    }
    g_rpc_transport_wsa_started = உண்மை;
    திரும்பு உண்மை;
#இல்லையெனில்
    திரும்பு உண்மை;
#endif
}

வெறுமை rpc_transport_shutdown() {
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_rpc_transport_mu);
    எனில் (!g_rpc_transport_wsa_started) {
        திரும்பு;
    }
    WSACleanup();
    g_rpc_transport_wsa_started = பொய்;
#endif
}

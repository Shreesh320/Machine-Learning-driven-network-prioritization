from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, arp, ipv4, tcp, udp
import time, csv, os

CSV_FILE = "flow_ml_simple.csv"
POLL_INTERVAL = 5
MAX_THR = 50 * 10**9

class FlowPollerSimple(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FlowPollerSimple, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.prev = {}          
        self.mac2ip = {}        
        self.flow_ports = {}    
        self._init_csv()
        hub.spawn(self._poller)
        self.logger.info("FlowPoller started -> %s", os.path.abspath(CSV_FILE))

    def _init_csv(self):
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp","dpid","table_id","in_port",
                    "eth_src","eth_dst","ip_src","ip_dst",
                    "eth_type","ip_proto","src_port","dst_port",
                    "n_packets","n_bytes","duration_sec",
                    "entry_type","throughput_bps","priority_label"
                ])

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            if dp.id in self.datapaths:
                del self.datapaths[dp.id]

    def _poller(self):
        while True:
            for dp in list(self.datapaths.values()):
                req = dp.ofproto_parser.OFPFlowStatsRequest(dp)
                dp.send_msg(req)
            hub.sleep(POLL_INTERVAL)

    # PacketIn learning
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _pkt_in(self, ev):
        msg = ev.msg
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if not eth:
            return

        src = eth.src
        dst = eth.dst

        # ARP/IP learning
        arp_pkt = pkt.get_protocol(arp.arp)
        ip_pkt  = pkt.get_protocol(ipv4.ipv4)

        if arp_pkt:
            self.mac2ip[src] = arp_pkt.src_ip

        if ip_pkt:
            self.mac2ip[src] = ip_pkt.src

        # Learn transport ports
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

        if tcp_pkt or udp_pkt:
            sport = tcp_pkt.src_port if tcp_pkt else udp_pkt.src_port
            dport = tcp_pkt.dst_port if tcp_pkt else udp_pkt.dst_port
            self.flow_ports[(src, dst)] = (sport, dport)
            if ip_pkt:
                self.flow_ports[(ip_pkt.src, ip_pkt.dst)] = (sport, dport)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats(self, ev):
        dp = ev.msg.datapath
        now = time.time()
        rows = []

        for stat in ev.msg.body:
            m = stat.match
            if not m or not m.fields:
                continue 

            eth_src = m.get("eth_src")
            eth_dst = m.get("eth_dst")
            if not eth_src or not eth_dst:
                continue

            ip_src = m.get("ipv4_src") or self.mac2ip.get(eth_src, "N/A")
            ip_dst = m.get("ipv4_dst") or self.mac2ip.get(eth_dst, "N/A")

            ip_proto = m.get("ip_proto", "N/A")
            in_port  = m.get("in_port", "N/A")

            tcp_src = m.get("tcp_src")
            tcp_dst = m.get("tcp_dst")
            udp_src = m.get("udp_src")
            udp_dst = m.get("udp_dst")

            src_port = tcp_src or udp_src
            dst_port = tcp_dst or udp_dst

            # fill transport ports using learned state
            if not src_port or not dst_port:
                if (eth_src, eth_dst) in self.flow_ports:
                    src_port, dst_port = self.flow_ports[(eth_src, eth_dst)]

            src_port = src_port or "N/A"
            dst_port = dst_port or "N/A"

            n_pkts = stat.packet_count
            n_byt  = stat.byte_count
            dur    = stat.duration_sec or 1

            if n_pkts == 0 and n_byt == 0:
                continue

            # throughput
            key = (dp.id, str(m))
            prev = self.prev.get(key)

            if prev:
                p_bytes, p_ts = prev
                dt = now - p_ts
                delta = n_byt - p_bytes
                thr = (delta * 8 / dt) if dt > 0 and delta > 0 else (n_byt * 8 / dur)
            else:
                thr = n_byt * 8 / dur

            thr = min(MAX_THR, thr)
            label = 1 if thr > 1_000_000 else 0

            rows.append([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                dp.id, stat.table_id, in_port,
                eth_src, eth_dst, ip_src, ip_dst,
                m.get("eth_type"), ip_proto,
                src_port, dst_port,
                n_pkts, n_byt, dur,
                "flow_stats", round(thr, 2), label
            ])

            self.prev[key] = (n_byt, now)

        if rows:
            with open(CSV_FILE, "a", newline='') as f:
                csv.writer(f).writerows(rows)
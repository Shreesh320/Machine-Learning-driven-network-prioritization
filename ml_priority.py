from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
import pandas as pd
import joblib
import time

MODEL_PATH = "/home/shreesh320/sdn-lab/flow_priority_model.pkl"

class MLPriority(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MLPriority, self).__init__(*args, **kwargs)
        self.model = joblib.load(MODEL_PATH)
        self.logger.info("ML model loaded.")
        self.mac_to_port = {}

    # install table-miss
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto

        match = parser.OFPMatch()
        inst = [parser.OFPInstructionActions(
            ofp.OFPIT_APPLY_ACTIONS,
            [parser.OFPActionOutput(ofp.OFPP_CONTROLLER)]
        )]

        dp.send_msg(parser.OFPFlowMod(
            datapath=dp, priority=0, match=match, instructions=inst
        ))

    # main packet_in
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        src = eth.src
        dst = eth.dst

        # learn
        self.mac_to_port.setdefault(dp.id, {})
        self.mac_to_port[dp.id][src] = in_port

        # simple L2 switching
        if dst in self.mac_to_port[dp.id]:
            out_port = self.mac_to_port[dp.id][dst]
        else:
            out_port = ofp.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # ML features (same as training set)
        features = {
            "n_packets": msg.total_len,
            "n_bytes": msg.total_len,
            "duration_sec": 1,
            "throughput_bps": msg.total_len * 8,
            "ip_proto": msg.match.get("ip_proto", 0),
            "src_port": msg.match.get("tcp_src", msg.match.get("udp_src", 0)),
            "dst_port": msg.match.get("tcp_dst", msg.match.get("udp_dst", 0)),
            "bytes_per_pkt": msg.total_len,
            "is_tcp": 1 if msg.match.get("ip_proto") == 6 else 0,
            "is_udp": 1 if msg.match.get("ip_proto") == 17 else 0
        }

        df = pd.DataFrame([features])
        try:
            pred = int(self.model.predict(df)[0])
        except:
            pred = 0

        priority = 100 if pred == 1 else 10

        # install ML-based flow rule
        match = parser.OFPMatch(eth_src=src, eth_dst=dst)
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]

        dp.send_msg(parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=10,
            hard_timeout=30
        ))

        # PacketOut to continue forwarding
        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        )
        dp.send_msg(out)
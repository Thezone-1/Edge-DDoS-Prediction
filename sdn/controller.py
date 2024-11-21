from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_0
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import ipv4, packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from DrDos_model import DDoSTransformer

import torch
import numpy as np
import ipaddress

# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import ipaddress


class ModelInference:
    def __init__(self, model_path, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = DDoSTransformer(input_dim=3)

        # Load the model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_data):
        # Ensure input is a tensor and on the correct device
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = input_data.to(self.device)

        # Add batch dimension if needed
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_data)

        return output


class Switch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Switch, self).__init__(*args, **kwargs)
        self.mac_to_port = {}

    def add_flow(self, datapath, in_port, dst, src, actions):
        ofproto = datapath.ofproto

        match = datapath.ofproto_parser.OFPMatch(
            in_port=in_port, dl_dst=haddr_to_bin(dst), dl_src=haddr_to_bin(src)
        )

        mod = datapath.ofproto_parser.OFPFlowMod(
            datapath=datapath,
            match=match,
            cookie=0,
            command=ofproto.OFPFC_ADD,
            idle_timeout=0,
            hard_timeout=0,
            priority=ofproto.OFP_DEFAULT_PRIORITY,
            flags=ofproto.OFPFF_SEND_FLOW_REM,
            actions=actions,
        )
        datapath.send_msg(mod)

    def ip_to_float(self, ip):
        """
        Convert an IPv4 address to a normalized float.

        :param ip: IPv4 address in string form.
        :return: Normalized float representation of the IP address.
        """
        # Convert IP address to integer
        ip_int = int(ipaddress.IPv4Address(ip))

        # Normalize to range [0, 1]
        max_int = int(ipaddress.IPv4Address("255.255.255.255"))
        return ip_int / max_int

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return

        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        if ipv4_pkt:
            dst_ip = ipv4_pkt.dst
            src_ip = ipv4_pkt.src
            dst_ip = self.ip_to_float(dst_ip)
            src_ip = self.ip_to_float(src_ip)
            total_length = ipv4_pkt.total_length

            input_data = np.array([dst_ip, src_ip, total_length])
        print("PACKET", ipv4_pkt)
        print("INPUT DATA", input_data)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        model_path = "best_model.pth"
        inference = ModelInference(model_path, self.device)
        print("INFERENCE", inference.predict(input_data))

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.logger.info("packet in %s %s %s %s", dpid, src, dst, msg.in_port)

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = msg.in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [datapath.ofproto_parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            self.add_flow(datapath, msg.in_port, dst, src, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = datapath.ofproto_parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=msg.in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)

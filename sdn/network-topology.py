from mininet.net import Mininet
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.node import RemoteController


RYU_IP = "127.0.0.1"
RYU_PORT = 6653

def customNet():
    net = Mininet(topo=None, build=False)

    info("Adding Ryu controller\n")
    net.addController(
        "c0",
        controller=RemoteController,
        ip=RYU_IP,
        port=RYU_PORT,
    )

    info("Adding hosts\n")
    h1, h2, h3 = [net.addHost(h) for h in ("h1", "h2", "h3")]

    info("Adding switches\n")
    s1 = net.addSwitch("s1")

    info("Adding switch links\n")
    for h in [h1, h2, h3]:
        net.addLink(h, s1)

    info("Starting network\n")
    net.start()

    CLI(net)

    info("Stopping network\n")
    net.stop()

if __name__ == "__main__":
    setLogLevel("info")
    customNet()


extends Node

var udp := PacketPeerUDP.new()
var json := JSON.new()

func _ready():
    var err = udp.listen(12000, "127.0.0.1")
    if err != OK:
        push_error("UDP listen failed: %s" % err)
    print("Listening on UDP 12000")

func _process(_dt):
    while udp.get_available_packet_count() > 0:
        var pkt = udp.get_packet().get_string_from_utf8()
        var data = JSON.parse_string(pkt)
        if typeof(data) == TYPE_DICTIONARY:
            # Example: alpha band average across channels
            var alpha = data["features"]["alpha"]
            var avg_alpha = 0.0
            for v in alpha:
                avg_alpha += v
            avg_alpha /= max(1, alpha.size())
            # use avg_alpha in your game

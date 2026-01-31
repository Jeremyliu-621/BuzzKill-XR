extends Control

@onready var bar: ProgressBar = $ProgressBar
var udp := PacketPeerUDP.new()
var port := 5005

func _ready():
	var err = udp.bind(port, "127.0.0.1")
	if err != OK:
		print("UDP bind failed:", err)
	else:
		print("Listening on UDP port", port)

func _process(_delta):
	while udp.get_available_packet_count() > 0:
		var text = udp.get_packet().get_string_from_utf8().strip_edges()
		var value = clamp(float(text), 0.0, 1.0)
		bar.value = value

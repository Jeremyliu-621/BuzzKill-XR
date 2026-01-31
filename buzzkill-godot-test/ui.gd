extends Control

@onready var bar: ProgressBar = $ProgressBar
var udp := PacketPeerUDP.new()
var json := JSON.new()
var port := 12000

# Normalization parameters for alpha band power
var min_alpha := 0.0
var max_alpha := 100.0  # Adjust based on your Muse's typical range
var alpha_history := []  # For smoothing/normalization

func _ready():
	var err = udp.bind(port, "127.0.0.1")
	if err != OK:
		push_error("UDP bind failed: %s" % err)
		print("UDP bind failed:", err)
	else:
		print("Listening on UDP port", port, " for Muse LSL data")

func _process(_delta):
	while udp.get_available_packet_count() > 0:
		var pkt = udp.get_packet()
		var pkt_string = pkt.get_string_from_utf8()
		var data = json.parse_string(pkt_string)
		
		if typeof(data) == TYPE_DICTIONARY and data.has("features"):
			var features = data["features"]
			
			# Extract alpha band values (commonly associated with concentration/relaxation)
			if features.has("alpha") and features["alpha"].size() > 0:
				var alpha_values = features["alpha"]
				var avg_alpha = 0.0
				for v in alpha_values:
					avg_alpha += v
				avg_alpha /= max(1, alpha_values.size())
				
				# Update min/max for dynamic normalization
				if alpha_history.size() < 100:  # Keep last 100 samples
					alpha_history.append(avg_alpha)
				else:
					alpha_history.pop_front()
					alpha_history.append(avg_alpha)
				
				if alpha_history.size() > 10:
					min_alpha = alpha_history.min()
					max_alpha = alpha_history.max()
				
				# Normalize to 0.0-1.0 range
				var range_alpha = max_alpha - min_alpha
				var normalized_value = 0.5  # Default to middle
				if range_alpha > 0.001:  # Avoid division by zero
					normalized_value = (avg_alpha - min_alpha) / range_alpha
				
				# Clamp and update progress bar
				normalized_value = clamp(normalized_value, 0.0, 1.0)
				bar.value = normalized_value

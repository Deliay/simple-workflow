# Name two downloads with the {file=...} tag on the node that produces them and
# bundle them into a single zip. The tag supplies each member's filename inside
# the archive; a zip input without a {file=...} tag is an error.
a[bv:BV1TqaR67E7f]{file=测试.mp4}
b[bv:BV2xxxxxxx]{file=测试2.mp4}
[var:a, var:b] -> [zip] -> [final]

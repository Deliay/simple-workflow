# Tag two downloads with [file:name] (a terminal node) and bundle them into a
# single zip. Each [file:name] tag supplies the member's filename inside the
# archive; a zip input without a tag is an error.
a[bv:BV1TqaR67E7f] -> [file:测试.mp4]
b[bv:BV2xxxxxxx] -> [file:测试2.mp4]
[var:a, var:b] -> [zip] -> [final]

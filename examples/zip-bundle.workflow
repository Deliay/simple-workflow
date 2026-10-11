# Bundle two downloads into one zip. Each {file=...} tag names a member.

# Inline: tag each producer and group them in an outer [...] bundle.
[[bv:BV1TqaR67E7f]{file=测试.mp4}, [bv:BV2xxxxxxx]{file=测试2.mp4}] -> [zip] -> [final]

# Equivalent, naming the intermediate values and referencing them:
# a[bv:BV1TqaR67E7f]{file=测试.mp4}
# b[bv:BV2xxxxxxx]{file=测试2.mp4}
# [var:a, var:b] -> [zip] -> [final]

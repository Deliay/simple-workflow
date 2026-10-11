# Vocal2Midi: vocal audio -> zip of mid/ustx/txt artifacts (language forced, e.g. zh).
[input:vocal] -> [v2m:zh] -> [final]

# DiffSinger (SVS): render an OpenUtau .ustx score with a voicebank -> wav.
[input:score] -> rendered[svs:小松绿v1.1]
[var:rendered] -> [final]

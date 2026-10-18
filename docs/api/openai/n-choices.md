# n > 1 choices

`n` runs that many generations (1..8, emulated sequentially, with the counters
summed): each emulated choice is a full generation, so `n` outside 1..8 or
non-integer is rejected with a `400`.

Streaming responses carry per-choice indexes; the summed `usage` matches the
non-streaming shape.

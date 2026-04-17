# Agent Notes

Bob - Original fixed-map MC PPO
**Fred** - Good MC PPO
Walt - MC GAE, really really slow -> constant
Marv - MC GAE, huge network still, slow -> constant
MarvJr - MC GAE MB, normalized, smoothed, big network -> constant
Larry - Long MC GAE MB PPO, normalized, smooth, smaller network -> constant
*Paul* - Fred but GAE -> constant
*James* - Paul with Normalization and different seed -> constant
*Tom* - James with ClipEpsilon = 0.4 -> no learning after 1k batches
*John* - James with no exploration bonus
*Jim* - Paul with no exploration bonus (so no normalization)
**Jeremy** - Fred but GAE (finally fixed shape bug....)
Rod - Jeremy but with shorter truncation limit -> collapsed to wait
Todd - Jeremy with extra reward shaping and shorter truncation limit. -> collapsed to wait
Ned - Todd with higher entropy bonus (0.07)

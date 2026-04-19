# Agent Notes

Key:
**good**
*success but bad*


Agents:
- Bob - Original fixed-map MC PPO
- **Fred** - Good MC PPO -> High variance
- Walt - GAE PPO, really really slow -> constant learning (shape bug!)
- Marv - GAE PPO, huge network still, slow -> constant learning (shape bug!)
- MarvJr - GAE PPO MB, normalized, smoothed, big network -> constant learning (shape bug!)
- Larry - Long GAE PPO MB PPO, normalized, smooth, smaller network -> constant learning (shape bug!)
- Paul - Fred but GAE -> constant learning (shape bug!)
- James - Paul with Normalization and different seed -> constant learning (shape bug!)
- Tom - James with ClipEpsilon = 0.4 -> no learning after 1k batches (shape bug!)
- John - James with no exploration bonus (shape bug!)
- Jim - Paul with no exploration bonus (so no normalization) (shape bug!)
- **Jeremy** - Fred but GAE (finally fixed shape bug....) -> Amazing
- *Rod* - Jeremy but with shorter truncation limit -> collapsed to wait
- *Todd* - Jeremy with extra reward shaping and shorter truncation limit. -> collapsed to wait
- *Ned* - Todd with higher entropy bonus (0.07) -> collapsed to wait (more slowly)

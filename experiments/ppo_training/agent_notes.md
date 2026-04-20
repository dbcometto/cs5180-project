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




## Theoretical Max Score:
10x15xREWARD_EXPLORE_TILE(0.2) = 30
10x15xREWARD_NEW_3D(0.05) = 7.5
6x4xREWARD_NEW_FACE(0.8) = 19.2
REWARD_COMPLETE(20) = 20
= 76.7
(excepting step and scan penalties)


Realistically,
60xREWARD_STEP(-0.05) = -3
12xREWARD_SCAN(-0.7) = -8.4
= 68.3

### Other Reward Params
REWARD_FAIL = -20
REWARD_FAR_END = -5
REWARD_PERCENT_MAX = 10
REWARD_MAX_EARLY_END = -10
REWARD_INVALID_END = -0.5
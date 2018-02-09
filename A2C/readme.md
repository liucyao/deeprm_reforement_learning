# A2C advantage actor-critic
- A3C 中的 Asynchronous 是导入异步训练的思想，但是这个算法只增强了CPU的计算能力，没有扩展到GPU上。所以我没有使用 Asynchronous，仅仅使用GPU版的 advantage actor-critic，即A2C。
- A3C 虽然没有利用GPU，但是nivida意识到了这点，推出了CPU/GPU混合实现的A3C——[NVlabs/GA3C](https://github.com/NVlabs/GA3C)，这个以后有机会在尝试吧。;)

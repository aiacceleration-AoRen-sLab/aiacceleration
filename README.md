# Large Model Compression & Intelligent Terminal Acceleration

# 大模型压缩与智能终端加速实验室

本仓库汇集了**大模型压缩与智能终端加速**实验室在模型轻量化领域的最新研究成果。我们致力于解决大模型（LLMs）、视觉Transformer（ViTs）及多模态大模型（LMMs）在资源受限终端上的部署难题。

## 1. 概况 (Overview)

随着深度学习模型参数量的爆炸式增长，计算和存储需求成为阻碍其在边缘设备普及的主要瓶颈。本实验室专注于后训练压缩（Post-Training Compression）技术，旨在无需昂贵重训练的情况下，显著降低模型资源消耗。

------

## 2. 非结构化剪枝 (Unstructured Pruning)

非结构化剪枝通过移除权重矩阵中的单个冗余元素来实现压缩。针对当前后训练剪枝（PTP）中存在的激活分布漂移和注意力机制长尾分布破坏等核心问题，我们深入研究了剪枝算法与动态稀疏度分配策略。

### 2.1 剪枝算法 (Pruning Algorithms)

#### 2.1.1 常规方法 (Conventional Methods)

现有的后训练剪枝算法主要分为“非权重更新”和“权重更新”两类，尽管它们取得了一定进展，但仍存在本质局限性：

- **非权重更新方法 (Non-Weight-Update Methods)**

  - **代表算法**：Magnitude Pruning, Wanda , Pruner-Zero.

    

    

  -  **原理**：这类方法定义一个剪枝度量标准（如权重大小与输入激活范数的乘积）来衡量权重重要性，直接移除不重要的权重而不更新剩余权重 。

    

    

  - **局限性**：Wanda 和 Pruner-Zero 等方法使用的度量标准本质上反映的是输出幅度而非真实的剪枝误差 。此外，它们完全保留原始权重值的做法在注意力模块中会导致误差累积，无法有效保持长尾分布特性 。

    

    

- **权重更新方法 (Weight-Update Methods)**

  -  **代表算法**：SparseGPT , SparseLLM , ADMM-Grad.

    

    

  -  **原理**：基于 Optimal Brain Surgeon (OBS) 框架，利用海森矩阵（Hessian）信息在剪枝后更新剩余权重，以最小化层级输出误差 。

    

    

  - **局限性**：

    1. **恒定激活假设 (Constant Activation Assumption)**：主流方法（如 SparseGPT）假设网络各层的输入激活在剪枝过程中是固定的 。然而，现代大模型在不同任务间存在显著的激活分布漂移（Activation Drift），忽略这种漂移会导致误差估计不准 。

       

       

    2. **忽视长尾分布 (Neglect of Long-tailed Attention)**：Transformer 的多头注意力（MHA）机制具有显著的长尾分布特征，少数 Token 主导了注意力分数 。现有的权重更新方法对所有 Q/K/V 权重进行无差别的更新，极易破坏这种关键的语义聚焦模式，导致注意力分布均一化和性能下降 。

       

       

#### 2.1.2 我们的方法 (Our Methods: D2Prune & D2ADMM)

为了解决上述问题，我们提出了统一的激活与注意力感知剪枝框架 D2Prune 及其增强版 D2Prune++。

- **双泰勒展开机制 (Dual-Taylor Expansion)** 针对“恒定激活假设”的缺陷，我们引入了双泰勒展开来指导掩码选择和权重更新 。

  

  

  -**原理**：我们将重构误差建模为权重和激活的双变量函数，联合捕捉权重扰动和激活漂移的影响 。

    

    

  - **公式**：误差变化被近似为：

    $$\delta E \approx \lambda_1 y w^T x + \lambda_2 x^T H_{22} x + \frac{1}{2} \delta w^T H_{11} \delta w$$

    其中 $\lambda$ 系数捕捉激活漂移的一阶和二阶效应 。这使得我们能够在激活波动的情况下更忠实地估计敏感度 。

    

    

- **注意力分布感知动态更新 (Attention Distribution-Aware Dynamic Update)** 针对长尾分布被破坏的问题，我们设计了一种动态权重更新策略 。

  

  

  -**机制**：我们将 Q/K/V 的更新状态建模为一个组合优化问题，在最小化重构误差的同时，引入 KL 散度约束来保持注意力分布的原始形态 。

    

    

  -**效果**：该策略能够选择性地更新 Q/K/V 投影，在补偿剪枝误差的同时，保留关键的长尾注意力模式 。

    

    

- **D2ADMM 全局权重恢复 (D2ADMM Global Recovery)** 为了进一步提升高稀疏度下的鲁棒性，我们在 D2Prune++ 中提出了 D2ADMM 模块 。

  

  

  -**全局协调**：不同于 SparseGPT 的局部贪婪更新，D2ADMM 利用交替方向乘子法（ADMM）在掩码固定后进行全局协调的权重恢复 。

    

    

  -**激活感知预缩放 (Activation-Aware Pre-scaling)**：我们引入预缩放机制，将双泰勒度量融入 ADMM 优化景观中，通过归一化激活统计数据来增强对激活漂移的鲁棒性 。

    

    

### 2.2 动态稀疏度方法 (Dynamic Sparsity Methods)

#### 2.2.1 常规方法 (Conventional Methods)

大多数现有工作采用**均匀稀疏度（Uniform Sparsity）**策略，即对所有层应用相同的剪枝率 。然而，不同层对剪枝的敏感度差异巨大，均匀分配往往导致次优的性能 。



-**基于规则/度量的方法**：如 **OWL** ，根据 Wanda 风格的度量计算层级离群值比例来分配稀疏度。这类方法虽然无需训练，但在高稀疏度下很难找到最优配置，甚至导致性能下降 。

  

  

-**基于梯度的学习方法**：如 **BESA** ，通过反向传播学习稀疏度参数。这类方法虽然效果较好，但需要昂贵的计算资源和显存开销，违背了后训练剪枝高效、低成本的初衷 。

  

  

#### 2.2.2 我们的方法 (Our Method: ZODSA)

我们提出了 **ZODSA (Zero-Order Dynamic Sparsity Allocation)**，一种内存高效、无梯度的块级动态稀疏度分配方法 。



- **无梯度优化 (Gradient-Free Optimization)** 受 MeZO 启发，ZODSA 将层级稀疏度分配重构为连续优化问题，利用零阶估计器仅通过两次前向传播即可估算梯度 。这完全消除了反向传播的需求，极大降低了内存占用 。

  

  

- **切空间投影 (Tangent-Space Projection)** 为了在优化过程中严格遵守全局稀疏度约束（即总参数量不变），我们构建了一个投影算子 $P$，将搜索方向限制在可行域的切空间内 ：

  

  

  $$P = I - \frac{NN^T}{||N||_2^2}$$

  这确保了每次更新都在满足全局压缩目标的前提下，灵活地在层间重新分配稀疏度 。

  

  

- **基于离群值的热启动 (Outlier-Aware Warm-Start)** 利用激活离群值作为结构先验来初始化搜索，ZODSA 能够比随机或均匀初始化更快收敛到高质量的稀疏配置 。

  

  

------

## 3. 视觉图像剪枝 (Vision Image Pruning)

*(本部分将展示我们在 Vision Transformers (ViTs) 上的剪枝成果，框架保留)*

------

## 4. 结构与半结构化剪枝 (Structured & Semi-Structured Pruning)

*(本部分将展示我们在 N:M 半结构化剪枝方面的研究，框架保留)*

------

## 5. 量化 (Quantization)

*(本部分将介绍实验室在模型量化方面的相关工作，框架保留)*

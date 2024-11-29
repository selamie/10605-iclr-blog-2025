---
layout: distill
title: The Frontier of Pipeline Parallelism
description: Comparing approaches to resolving the pareto frontier in parallel model training at scale
date: 2024-11-29
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Selam Gano
    url: ""
    affiliations:
      name: CMU
  - name: Owen Li
    url: ""
    affiliations:
      name: CMU

# must be the exact same name as your blogpost
bibliography: 2025-04-28-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.

# toc:
#   - name: Equations
#   - name: Images and Figures
#     subsections:
#     - name: Interactive Figures
#   - name: Citations
#   - name: Footnotes
#   - name: Code Blocks
#   - name: Diagrams
#   - name: Tweets
#   - name: Layouts
#   - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.

# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---

## Motivation: Deep Learning and Distributed Systems 

In modern machine learning, large deep neural networks with billions of parameters are run on distributed systems, with a single model being trained on many different machines. This presents a problem for model developers, who have to design machine learning systems for distributed training. 

Training modern large-scale models requires effective techniques for partitioning and parallelizing computation. This problem ultimately results in three key traits to optimize for: 
  - **how long** the model will take to train
  - **how much memory** it will require, and 
  - how **efficiently** the available accelerators (i.e. GPUs) will be used. 

The efficiency of accelerator usage is related both to how long the model takes to train and how many resources may be required for a given model size. 

Multiple types of parallelization techniques have emerged as model sizes and distributed training architectures have grown. 

## Parallel Model Training: Data vs Model Parallelism

**Data parallelism** helps to parallelize computation by creating copies of the model on each accelerator, then partitioning, or "sharding", the data and distributing it to the different devices. The results must be aggregated for the backwards propagation step. This can help to distribute a small enough model over multiple GPUs, but cannot address the case when the model itself is too large to fit on a single GPU. 

<!-- note: in multiple papers (survey paper, zero bubble paper) data parallelism is described this way,
 so I continued using this definition -->

In **model parallelism**, the model itself is divided into multiple partitions of the original model, then allocated to different devices. There are two ways to partition a model--"horizontally" and "vertically". 

{% include figure.html path="assets/img/parallelism_strategies_overview.svg" class="img-fluid" %}
<div class="caption">
  Image source: Phillip Lippe <d-cite key="lippe2024uvadlc"></d-cite> 
</div>

<!-- cite image: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_simple.html -->

The horizontal approach, "tensor parallelism" or intra-layer parallelism, splits each tensor in the model into different chunks, and during processing results are synchronized at each step. The vertical approach, **"pipeline parallelism"** or inter-layer parallelism, segments the model into stages, like a pipeline, and only the resulting activation values and gradients need to be transmitted between GPUs. Tensor parallelism generally has higher communication overhead than pipeline parallelism because all the results of these partial tensor computations must be transmitted. 

<!-- However, [it can potentially be more memory efficient and reduce GPU idle time](https://www.determined.ai/blog/tp).  -->

Of course, it is possible to use the idea of sharding data from data parallelism in addition to partitioning the model, known as **hybrid parallelism**. Recent approaches use many specific techniques to try to improve overall efficiency. Though it is unrealistic to absolutely minimize all three of **storage, computation, and communication** costs, a given approach will try to find the best tradeoff between them by attacking each of these issues with a specific technique. 

### Naive Model Parallelism, and a Problem
Consider the process of training a model consisting of $p$ fully connected layers: there is a sequence of forward steps $f_1, ..., f_p$, followed by a sequence of back-propagation steps $b_p, b_{p-1}, ..., b_1$. If the size of the model is too big, it may be necessary to spread the training task among several devices. In particular, since each pair of forward and backward passes may use the same data, like the weight $W$ of the fully-connected neural network layer, a natural choice is to designate a single "device" to handle both forward and backward passes of the same stage. 

Naively, given computational devices $d_1, ..., d_p$, one might accomplish such a task in the following manner:
```
1. Use d_1 to compute x_1 = sigma(W_1 @ x_0)
2. Use d_2 to compute x_2 = sigma(W_2 @ x_1 )
.
.
.
p. Use d_p to compute y = sigma(W_p @ x_{p-1});
   Obtain gradient from loss function associated with y.
p+1. Use d_p to compute gradient L_p using gradient of loss and W_p,
     then update W_p.
p+2. Use d_{p-1} to compute gradient L_{p-1} using L_p and W_{p-1}, 
     then update W_{p-1}.
.
.
.
2p. Use d_1 to compute gradient L_1, and update W_1
```
If we were to create a plot representing what each of the devices are doing at any time, it would look like the following: 

{% include figure.html path="assets/img/naive_placeholder.png" class="img-fluid" %}
<div class="caption"> 
  Image Source: Guan et. al.  <d-cite key="guan2024advances"></d-cite> 
</div>

Notice how each device (ie. GPU) is idle for most of the time? If we were to consider a very simplistic case where each task, whether it is forward or backward computation, takes exactly 1 second to complete, then each device is only working for $2$ out of the $2p$ seconds, meaning a lot of hardware is sitting around, doing nothing. This idling is indicated by the white spaces in the plot, known as **bubbles**. The term **bubble ratio** is used to describe the ratio of such hardware idle time. 

Evidently, having a high bubble ratio is inefficient usage of computing resources. By allocating computing tasks on GPU properly, we may make hardware usage or model training more efficient; such allocation strategies are called **Pipeline Scheduling**, which we discuss in the following section. 

### Pipeline Scheduling

In general, pipeline parallelism can be divided into **synchronous** and **asynchronous** scheduling approaches. The naive approach described is a **synchronous** approach, referring to the fact that, periodically, all model parameters are synchronously updated with the accumulated gradients at a given stage. This is the most **statistically efficient** approach as each update uses all learned information. 
As shown earlier in the example of naive pipeline parallelism, letting computing devices wait for the accumulated gradients lead to considerable device idling, and thus a high bubble ratio. 

**Asynchronous** approaches do not wait to update with all accumulated gradients, allowing higher GPU utilization and reduces bubble ratio. But this also means that later mini-batches in the pipeline may derive gradients with **stale weights**, which refers to weights that were computed in the past. This harms **statistical efficiency**, as we are updating the model weights using slightly outdated information. 

Given the tradeoffs between the two schedules, any particular implementation of each uses different techniques to try to improve the bubble ratio in the synchronous case, and the learning efficiency in the asynchronous case. Each attempt to mitigate these traits can come with memory and communication trade-offs in exchange for more efficient compute utilization or statistical efficiency. 

If synchronous scheduling is the most statistically efficient, why would you ever use an asynchronous schedule? Well, if the learning task were some ideal, parabolic convex optimization case, and moreover we have all the GPU's in the world, then it's possible there'd be no point to using any approach that harms the numerical progression down the true global minimum. But real machine learning scenarios typically have the following traits: 
- Objective function is messier, so in most cases, we measure the model's actual performance on a problem, and accurate convergence to global min may not be the biggest concern, and 
- We do not have all the GPU's in the world, so having some of them staying idle during model training is not economical. 
  
Essentially, **it may be advantageous to get to a less-than-ideal convergence faster while making better utilization of our available compute resources**, especially if "faster" is the difference between practically possible or not. 

## Practical Approaches to Pipeline Parallelism

Now, we'll look at several different pipeline parallelism approaches and how they choose to mitigate  different tradeoffs. As a reminder, in the ideal case, we'd like to achieve: 
  - minimal memory consumption 
  - low communication overhead
  - efficient compute utilization
    - For synchronous approaches, we would like to reduce the bubble ratio
  - maximum statistical efficiency
    - For asynchronous approaches, we have to handle weight staleness and inconsistency

Finally, we also want an **even workload distributed across our GPUs**, or **load balance**, since one GPU working very hard and another not working that hard is not much better than one active GPU and one idle GPU. 

### GPipe: The Representative Synchronous Approach

One simple way to somewhat improve on this naive baseline is to **segment mini-batches of data** so that at least, during each forward and backward pass, each device can be working on a different segment of the mini-batch--or a **"micro-batch"**. This is how GPipe by Huang et. al. <d-cite key="huang2019gpipe"></d-cite>  tries to improve on the naive baseline, and it is representative of most practical synchronous approaches.  

- GPipe **communicates** only the output data of one model partition (possibly larger or smaller than a layer)
- GPipe **synchronizes gradients** and the optimizer step across micro-batches within a mini-batch
- The user specifies the number of micro-batches and the number of model partitions (equal to the number of available GPUs)
- GPipe **stores one version of weights total**

A direct consequence of GPipe's approach is higher peak memory required to store more activation values. This is somewhat mitigated by discarding and re-calculating activation values during the backward pass. However, re-calculation introduces a 20% increase in computation overhead <d-cite key="qi2024zero"></d-cite>. 

It shall also be noted that since GPipe is a synchronous approach, some computing devices may need to be waiting for the weight to synchronize, causing nontrivial bubble ratios.


### PipeDream: The Representative Asynchronous Approach

Pipedream by Narayanan et. al. <d-cite key="narayanan2019pipedream"></d-cite> is representative of the group of **asynchronous** approaches to pipeline parallelism, meaning that it occasionally must compute gradients of a mini-batch with "stale" weights, which can reduce learning efficiency (i.e., not every update has the same amount of information that it would in traditional model training)

- Pipedream **communicates** only the output data of a set of model layers (partitioned according to number of GPUs)
- Pipedream uses **asynchronous** computations of gradients
    - This technically reduces statistical efficiency since gradients can be computed on stale weights.
- Pipedream continuously calculates the number of optimal mini-batches at runtime.
- Pipedream **stores one version of weights per mini-batch**


{% include figure.html path="assets/img/1f1b.png" class="img-fluid" %}

<div class="caption"> 
  Image Source: Naranyan et. al.  <d-cite key="narayanan2019pipedream"></d-cite> 
</div>

Pipedream introduced the "interleaved 1F1B" or "one forward one backward" approach, where forward and backward passes of different micro-batches are interleaved to eliminate bubbles. However, PipeDream caches the weights used in the forward pass, all the way until these weights are used for computing gradient in the backward passes again, it has to store a lot of stale model weight. 

### PipeMare: Improved Asynchronous Approach
Pipemare by Yang et. al. <d-cite key="yang2021pipemare"></d-cite> improves the memory usage of PipeDream by approximating weights that appeared earlier in the pipeline, instead of caching them; then to ensure convergence, it also schedules learning rate accordingly. It strikes a perfect balance between GPipe and PipeDream, as it has virtually no bubble, and its memory usage is also relatively low.

The idea of PipeMare is to first do whatever PipeDream does, but then *simply use whatever weight $W$ in the memory to compute the gradient during backward step*, and avoid caching stale weights like PipeDream originally does. Since no device is waiting for a specific version of weights, there is no bubble; since there's no need to store older weights, the usage of memory is efficient. Both PipeDream and GPipe's issues are resolved!

Sounds intuitive, but what could go wrong?

Imaging training a stack of fully connected layers. Note that at the $k$th fully connected layer, computing gradient $g_k$ from backpropagation steps depends on two things: (1) some version of model weight stored in device $d_k$ , and (2) the loss at the model output layer, which depends on the forward pass at $k$th layer, processed by the same device $d_k$ at an earlier time, using an earlier version of model weights. In essence, the idea of PipeMare would result in computing the gradient using two different versions of model weights! We express this dependency as: 

$$w^+ = w - \nabla f(w_{\text{older}}, w_{\text{newer}})$$

This phenomenon is called **Delay Discrepancy**. In comparison, gradient descent is only defined using a fixed version of function input (ie. model weights), as below:

$$w^+ = w - \nabla f(w')$$

where for GPipe, $w' = w$, and for PipeDream, $w'$ stands for a single version of stale model weight. 

<!-- Note the trade off between GPipe and Pipedream: if we wish to update weights $w$ using only freshly computed weights, the device will sequentially wait for a certain weight $w_t$ at timestep $t$ to (1) go through forward pass; and then (2) go through back propagation path and produce an associated gradient, before computing the final update, which requires a long wait. 

PipeDream, on the other hand, updates weight $w$ using some version of weight $w_t$, and while waiting for $w_t$ to make its way all the way across forward and backward pass to produce $\nabla f_t$, the device can spend the time processing other weights. The problem is, while $w_t$ makes its round trip through forward-backward prop, all newer weight copies $w_{t+1}, w_{t+2}$... shall be stored to the device, before $w_t$ finally completes the round trip, be used to update the model weights, and gets removed, which is memory intensive.  -->
<!-- 
Now, PipeMare simultaneously resolved both the bubble ratio issue and to some extend. Since it doesn't wait for any weights, the idling bubble is small; since it doesn't store older weights, it is memory efficient.  -->

This discrepancy brings two issues: 

1. Since we are performing gradient descent using inconsistent versions of weights, would convergence be an issue: 
2. If $w^+ = w - \nabla f(w_{older}, w_{newer})$ then how do we know $w_{older}$ without caching it?  

PipeMare's novelty involves resolving these two problems separately: 

For **(1)**, we may locally approximate the objective function with $f(x) \approx \frac{\lambda}{2}x^2$ for simplicity; then the gradient update can be seen as   
<!-- $$w_{i+1} = w_i - \alpha \nabla f(...) = w_t - \alpha \lambda w_{t-\text{delay}} + \alpha \eta$$   
where $\alpha$ is the learning rate, ``delay'' is how long a particular version of gradient have delayed, and $\eta$ is the estimation of noise caused by asynchronous gradients. Then if we treat the collection of all versions of gradient over time as a single vector   
$$W_t = [w_t, w_{t-1}, ...]^\top$$  
then the gradient update rule can be written as some linear equation:   
$$W_{t+1} = CW_t + \alpha \eta e_1$$  
where $C$ is some suitable matrix, and $e_1$ is one-hot vector with first entry being 1. Convergence of gradient descent would hence depend on $C$'s eigenvalues, or equivalently, the roots of the characteristic polynomial 
$p(x) = x^{\text{delay}+1} - x^{\text{delay}} + \alpha \lambda$, to lie in the unit circle; solve for an appropriate $\alpha$ gives us a learning rate that lead to convergence. 

For **T2**, we once again locally approximate the objective function with $f(x) \approx \frac{\lambda}{2}x^2$ for simplicity; then we can can approximate the gradient update equation as    -->
$$w^+ = w - \nabla f(w_{older}, w_{newer}) \approx w - \lambda w_{newer} - \Delta (w_{newer} - w_{older}) + \eta $$  
where $\eta$ denotes a noise term and $\Delta$ is the sensitivity of $\nabla f$ to the discrepancy of gradient. Then if we treat the collection of all versions of gradient over time as a single vector   
$$W_t = [w_t, w_{t-1}, ...]^\top$$  
then the gradient update rule can be written as some linear equation:   
$$W_{t+1} = CW_t + \alpha \eta e_1$$  
where $C$ is some suitable matrix, and $e_1$ is one-hot vector with first entry being 1. Convergence of gradient descent can then be guaranteed by solving for a learning rate $\alpha$ that lets $C$'s eigenvalues stay in the unit ball, and hence the weight update rule would be stable. Specifically, <d-cite key="narayanan2019pipedream"></d-cite> shows that longer delays require smaller step sizes to ensure convergence. 

For **(2)**, we substitute approximation  
$$w_{older} \approx w_{newer} - \Delta\text{time}(w_{older}, w_{newer}) \cdot \delta$$  
into the linear equation mentioned earlier, where $\Delta\text{time}()$ denotes the difference in time-stamp, and $\delta$ is a trainable parameter that estimates how quickly model weights change over time. This allows estimating older weights using newer weights, eliminating the need of caching multiple version of stale weights, as PipeDream did.


### Zero-Bubble: An Improved Synchronous Approach

The Zero-Bubble (or "ZB") synchronous approach introduced by Qi et al. <d-cite key="qi2024zero"></d-cite> achieved  zero-bubble pipeline parallelism under synchronous scheduling. This was made possible by their innovation of **splitting up the gradient computation in the backward pass**, such that this, too, could be interleaved to eliminate bubbles. The approach demonstrates that grouping the backward pass calculations together sequential is unnecessary. 

{% include figure.html path="assets/img/ZB_Split.png" class="img-fluid" %}
<div class="caption"> 
  Image Source: Qi et. al.  <d-cite key="qi2024zero"></d-cite> 
</div>

There are two version of the ZB approach: ZB-H1, which consumes the same peak memory usage as 1F1B introduced by PipeDream <d-cite key="narayanan2019pipedream"></d-cite>, and ZB-H2, which eliminates bubbles completely but increases peak memory usage and has some extra computation needed. To eliminate bubbles completely, ZB-H2 also bypasses optimizer synchronization and instead introduces a validation and rollback step to rectify any miscalculations after the optimizer step (this is what results in some extra computation).

{% include figure.html path="assets/img/ZB-sched.png" class="img-fluid" %}
<div class="caption"> 
  ZB-H1 schedule (top) and ZB-H2 schedule (bottom) Image Source: Qi et. al.  <d-cite key="qi2024zero"></d-cite> 
</div>

{% include figure.html path="assets/img/rollback.png" class="img-fluid" %}
<div class="caption"> 
  Optimizer validation and rollback. Image Source: Qi et. al.  <d-cite key="qi2024zero"></d-cite> 
</div>

While ZB has a handcrafted schedule that works under the assumption that the execution times of the forward pass and each interleaved backward pass calculation are identical, an algorithm to automatically compose schedules is also introduced. The scheduling problem is formulated as integer linear programming that can be solved by an off-the-shelf ILP solver.

## Comparisons and Trade-offs
The previous section only discussed a few of these approaches in detail. However, if we broadly examine pipeline parallelism methods and their performance metrics in memory usage, computation resource utilization, and convergence, we see some interesting trade-offs. Below is a notation key and table comparing different approaches as compiled by Guan et. al. <d-cite key="guan2024advances"></d-cite>.

<!-- | Approach      | Schedule   | Bubble Ratio        | Convergence | |Extra Memory | Extra Compute | Extra Communication |  
| ------------- | ---------- | ------------------- | ----------- | |--------- | ---------- | ---------- |
| GPipe         | Synch      | $\frac{D}{D+T}$     |  Excellent  |           |            |
| GEMS          | Synch      |  $1 - \Theta(1/D)$  |  Excellent  | |X         |            |  X
| DAPPLE        | Synch      |  $\frac{D}{D+T}$    |  Excellent  | |          |            |
| Chimera       | Synch      |  $\frac{D}{D+2T}$   |  Excellent  | |X         |            | X
| Megatron-LM   | Synch      | $\frac{D}{vT}$      |  Excellent  | |          |            | X
| ZeroBubble*   | Synch      | $0$                 |  Excellent  |           | X          |  
| AMPNet        | Async      | $0$                 |  Poor       |           |            |
| PipeDream     | Async      | $0$                 |  Good       | X         |            |
| XPipe         | Async      | $0$                 |  Good       | X         | X          |
| SpecTrain     | Async      | $0$                 |  Good       | X         | X          |
| PipeDream-2BW | Async      | $0$                 |  Good       | X         |            |
| PipeMare      | Async      | $0$                 |  Good       | X         | X          |
| AvgPipe       | Async      | $0$                 |  Good       | X         |            |  X   
| WPipe         | Async      | $0$                 |  Good       | X         |            | -->

| Notation     | Description                                   |
|------------|-----------------------------------------------|
| $D$        | Number of pipeline stages                     |
| $P$        | Number of replicated Pipelines                |
| $B$        | Micro-batch size                              |
| $T$        | Number of micro-batches in each mini-batch    |
| $N$        | Mini-batch size ($N = T \times B$)            |
| $M_\theta$ | Memory consumption for weights of a stage     |
| $M_a$      | Memory consumption for activations of a stage |
| $v$        | number of chunks on each GPU                  |

| Approach      | Bubble Ratio                    | Convergence | Weights Memory                   | Activations Memory                    | Extra Computation Overhead |
|---------------|---------------------------------|-------------|----------------------------------|---------------------------------------|----------------------------|
| GPipe         | $(D - 1)/(T + D - 1)^*$         | Excellent   | $M_\theta$                       | $T \times M_a$                        |                            |
| GEMS          | $\approx (D - 1)/(D + 1/2)^*$   | Excellent   | $2M_\theta$                      | $M_a$                                 |                            |
| DAPPLE        | $(D - 1)/(D + T - 1)^*$         | Excellent   | $M_\theta$                       | $[M_a, D \times M_a]$                 |                            |
| Chimera       | $(D - 2)/(2T + D - 2)^*$        | Excellent   | $2M_\theta$                      | $[(D/2 + 1)M_a, D \times M_a]$        |                            |
| Megatron-LM   | $(D - 1)/(v \times T)$          | Excellent   | $M_\theta$                       | $T \times M_a$                        |                            |
| ZeroBubble (ZB-H2)| $\approx 0\%$               | Excellent   | $M_\theta$                       | $(2D - 1) \times M_a^\S$              | X                          |
| AMPNet        | $\approx 0\%$                   | Poor        | $M_\theta$                       | $[M_a, D \times M_a]$                 |                            |
| PipeDream     | $\approx 0\%$                   | Good        | $[M_\theta, D \times M_\theta]$  | $[M_a, D \times M_a]$                 |                            |
| XPipe         | $\approx 0\%$                   | Good        | $M_\theta$                       | $[M_a, D \times M_a]$                 | X                          |
| SpecTrain     | $\approx 0\%$                   | Good        | $M_\theta$                       | $[M_a, D \times M_a]$                 | X                          |
| PipeDream-2BW | $\approx 0\%$                   | Good        | $2M_\theta$                      | $[M_a, D \times M_a]$                 |                            |
| PipeMare      | $\approx 0\%$                   | Good        | $M_\theta$                       | $[M_a, D \times M_a]$                 | X                          |
| AvgPipe       | $\approx 0\%$                   | Good        | $P \times M_\theta$              | $[1, D \times T] \times P \times M_a$ |                            |
| WPipe         | $\approx 0\%$                   | Good        | $2M_\theta$                      | $T \times M_a$                        |                            |

<div class="caption"> 
  The data comprising this table is replicated from a survey by Guan et. al.  <d-cite key="guan2024advances"></d-cite>
</div>

Some of the techniques in this comparison table also require extra memory, computation, and communication overhead that do not scale directly with the included parameters. 

Observe that these approaches can be broadly categorized as follows:

1. Synchronous approaches, like GPipe and DAPPLE, which do not use stale weights for computing gradients; to wait for newest weights before computing gradients, the computing devices may need to stay idle, leading to nonzero bubble ratios, but due to being synchronous, they have high statistical efficiency and therefore excellent convergence. 
2. Asynchronous approaches that uses stale weights for gradient calculation, which allows devices to never go idle, and effectively eliminates bubble, but due to being asynchronous, they have relatively lower statistical efficiency, and therefore slightly worse convergence behavior. They can further be refined as: 
   1. Approaches that caches stale weights to handle weight discrepancy, like PipeDream, AvgPipe, and WPipe; these approaches don't need extra computation overhead, but always uses no less than $2M_\theta$ memory for storing weights. 
   2. Approaches that uses extra computation overhead to predict / approximate older versions stale weights, like Pipemare, XPipe, SpecTrain; these approaches only need to store one copy of model weights, so the weight memory usage is always $M_\theta$. 

Two notable exceptions include AMPNet, which neither caches extra copies of stale weights nor has computation overhead, but converges poorly; and ZeroBubble, whose unique approach is covered in earlier sections. 


## Conclusion and Discussion

Ultimately, for the ideal balance of zero bubbles in the pipeline, combined with synchronous training semantics, and a schedule that is optimally calculated with integer linear programming (ILP), ZeroBubble's ZB-H2 approach would be an ideal starting point and represent the current state of the art in pipeline parallelism. However, they note that the algorithm for computing ideal schedules may not scale well with an off-the-shelf ILP solver. This could lead to new approaches to optimize the algorithm, or construct ILP solutions specifically for machine learning applications. Like other trends in machine learning, such as [custom-built accelerators](https://cloud.google.com/tpu), pipeline parallelism could benefit from optimization tools specifically customized to machine learning applications. In addition, tolerance for additional memory and computation overhead should be assessed for each learning task when choosing which approach to apply. 
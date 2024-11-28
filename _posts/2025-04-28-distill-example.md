---
layout: distill
title: The Frontier of Pipeline Parallelism
description: Comparing approaches to resolving the pareto frontier in distributed deep learning
date: 2025-11-29
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

**Data parallelism** helps to parallelize computation by creating copies of the model on each accelerator, then partitioning, or "sharding", the data and distributing it to the different devices. The results must be aggregated for the backwards propogation step. This can help to distribute a small enough model over multiple GPUs, but cannot address the case when the model itself is too large to fit on a single GPU. 

<!-- note: in multiple papers (survey paper, zero bubble paper) data parallelism is described this way,
 so I continued using this definition -->

In **model parallelism**, the model itself is divided into multiple partitions of the original model, then allocated to different devices. There are two ways to partition a model--"horizontally" and "vertically". 

{% include figure.html path="assets/img/parallelism_strategies_overview.svg" class="img-fluid" %}
<!-- cite image: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/tensor_parallel_simple.html -->

The horizontal approach, "tensor parallelism" or intra-layer parallelism, splits each tensor in the model into different chunks, and during processing results are synchronized at each step. The vertical approach, **"pipeline parallelism"** or inter-layer parallelism, segments the model into stages, like a pipeline, and only the resulting activation values and gradients need to be transmitted between GPUs. 

Tensor parallelism generally has higher communication overhead than pipeline parallelism because all the results of these partial tensor computations must be transmitted, and inefficient AllReduce operations performed more frequently. However, [it can potentially be more memory efficient and reduce GPU idle time](https://www.determined.ai/blog/tp). 

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

Notice how each device (ie. GPU) is idle for most of the time? If we were to consider a very simplistic case where each task, whether it is forward or backward computation, takes exactly 1 second to complete, then each device is only working for $2$ out of the $2p$ seconds, meaning a lot of hardware is sitting around, doing nothing. This idling is indicated by the white spaces in the plot, and they are known as "Bubbles". In ML System literature, the term "Bubble Ratio" is used to describe the ratio of such hardware idle time. 

Evidently, letting most of the hardware idle is not ideal; However, there is a family of techniques that allows training model in a parallel manner, while reducing the ratio of such idle bubbles. We generally refer to such techniques as...

### Pipeline Scheduling

In general, pipeline parallelism can be divided into **synchronous** and **asynchronous** scheduling approaches. The naive approach described is a **synchronous** approach, referring to the fact that, periodically, all model parameters are synchronously updated with the accumulated gradients at a given stage. This is the most **statistically efficient** approach as each update uses all learned information. 

**Asynchronous** approaches do not wait to update with all accumulated gradients, allowing higher GPU utilization. But this also means that later mini-batches in the pipeline may derive gradients with stale weights, harming statistical efficiency. 

Given the tradeoffs between the two schedules, any particular implementation of each uses different techniques to try to improve the bubble ratio in the synchronous case, and the learning efficiency in the asynchronous case. Each attempt to mitigate these traits can come with memory and communication trade-offs in exchange for more efficient compute utilization or statistical efficiency. 

If synchronous scheduling is the most statistically efficient, why would you ever use an asynchronous schedule? Well, if the learning task were some ideal, parabolic convex optimization case, then it's possible there'd be no point to using any approach that harms the numerical progression down to some global minimum. But real machine learning problems are typically messier than this, and in most cases, rather than targeting an ideal convergence, we simply measure the model's actual performance on a problem. Essentially, **it may be advantageous to get to a less-than-ideal convergence faster with our available compute resources**, especially if "faster" is the difference between practically possible or not. 

## Practical Approaches to Pipeline Parallelism

Now, we'll look at several different pipeline parallelism approaches and how they choose to mitigate  different tradeoffs. As a reminder, in the ideal case, we'd like to acheive: 
  - minimal memory consumption 
  - low communication overhead
  - efficient compute utilization
    - For synchronous approaches, we would like to reduce the bubble ratio
  - maximum statistical efficiency
    - For asynchronous approahces, we have to handle weight staleness and inconsistency

Finally, we also want an **even workload distributed across our GPUs**, or **load balance**, since one GPU working very hard and another not working that hard is not much better than one active GPU and one idle GPU. 

<!-- [TO-DO]
(note to incorporate above:)
characteristics of pipeline parallelism approaches 
- synchronous vs asynchronous 
- data units used (mini-batch vs micro-batch)
-  

performance metrics:
- bubble sizes
- memory required

pipeline parallelism is reflective of other ml trends, creating different pipelines for different approaches(?) 

questions we should answer: 
- Why would you choose one approach vs another? 
- Is there any reason one approach could be better for a learning task vs another?
- Is the convergence column of survey paper correct *in practice*?  -->

### GPipe: The Representative Synchronous Approach

One simple way to somewhat improve on this naive baseline is to **segment mini-batches of data** so that at least, during each forward and backward pass, each device can be working on a different segment of the mini-batch--or a **"micro-batch"**. This is how GPipe tries to improve on the naive baseline, and it's representative of most practical synchronous approaches.  

<!-- insert fig -->

- GPipe **communicates** only the output data of one model partition (possibly larger or smaller than a layer)
- GPipe synchronizes gradients across micro-batches within a mini-batch
    - This leads to more idle time, or bubbles, for the system of GPUs.  
- The user specifies the number of micro-batches and the number of model partitions (equal to the number of available GPUs)
- GPipe stores one version of weights total

### PipeDream: The Representative Asynchronous Approach

<!-- PipeDream first introduced the term "pipeline parallelism" in a paper released in 2018. PipeDream systematically partitions DNN layers to keep all GPUs productive and minimize communication. It introduced the idea of pipelining minibatches in addition to partitioning layers of the model.  -->

Pipedream is representative of the group of **asynchronous** approaches to pipeline parallelism, meaning that it occasionally must compute gradients of a mini-batch with "stale" weights, which can reduce learning efficiency (i.e., not every update has the same amount of information that it would in traditional model training)

- Pipedream **communicates** only the output data of a set of model layers (partitioned according to number of GPUs)
- Pipedream uses **asynchronous** computations of gradients
    - This technically reduces statistical efficiency since gradients can be computed on stale weights.
- Pipedream continuously calculates the number of optimal minibatches at runtime.
- Pipedream stores one version of weights per mini-batch

### PipeMare
In one sentence, Pipemare conserves memory usage by approximating weights that appeared earlier in the pipeline, instead of caching them; then to ensure convergence, it also schedules learning rate accordingly. It strikes a perfect balance between GPipe and PipeDream. 

PipeDream uses the 1F1B mechanism to maintain a low bubble ratio, but since it computes the gradient by using the same weight in forward and backward passes, it has to store a lot of extra weight. 

PipeMare, on the other hand, simply uses whatever weight $W$ in the memory to compute the gradient, and does not use the cached historical weights. 

Sounds intuitive, but what could go wrong?

Note that at the $k$th fully connected NN layer, the computed gradient $g_k$ from backpropagation steps depends on two things: (1) the current weight stored in device $d_k$ , and (2) the loss at the model output layer, which depends on the forward pass at $k$th layer, processed by the same device $d_k$ at an earlier time, using an earlier version of model weights. In essence, it computes the gradient using two different versions of model weights! 

$$w^+ = w - \nabla f(w_{older}, w_{newer})$$

In comparison, gradient descent is only defined using a fixed version of function input (ie. model weights), as below:

$$w^+ = w - \nabla f(w')$$

where for GPipe, $w' = w$, and for PipeDream, $w'$ stands for a single version of gradient that is somewhat earlier than $w$. 

Note the trade off between GPipe and Pipedream: if we wish to update weights $w$ using only freshly computed weights, the device will sequentially wait for a certain weight $w_t$ at timestep $t$ to (1) go throught forward pass; and then (2) go through back propagation path and produce an associated gradient, before computing the final update, which requires a long wait. 

PipeDream, on the other hand, updates weight $w$ using some version of weight $w_t$, and while waiting for $w_t$ to make its way all the way across forward and backward pass to produce $\nabla f_t$, the device can spend the time processing other weights. The problem is, while $w_t$ makes its round trip throught forward-backward prop, all newer weight copies $w_{t+1}, w_{t+2}$... shall be stored to the device, before $w_t$ finally completes the round trip, be used to update the model weights, and gets removed, which is memory intensive. 

Now, Pipemare simultaneously resolved GPipe and PipeMare's issue to some extend. Since it doesn't wait for any weights, the idling bubble is small; since it doesn't store older weights, it is memory efficient. 

However, this brings two additional issues: 

1. Since we are performing gradient descent using inconsistent versions of weights, would convergence be an issue: 
2. If $w^+ = w - \nabla f(w_{older}, w_{newer})$ then how do we know $w_{older}$ without caching it?  

PipeMare resolves these two problems separately: 

For 1, we locally approximate the objective function with $f(x) \approx \frac{\lambda}{2}x^2$ for simplicity; then the gradient update can be seen as 
$$w_{i+1} = w_i - \alpha \nabla f(...) = w_t - \alpha \lambda w_{t-\text{delay}} + \alpha \eta$$ 
where $\alpha$ is the learning rate, ``delay'' is how long a particular version of gradient have delayed, and $\eta$ is the estimation of noise caused by asynchronous gradients. Then if we treat the collection of all versions of gradient over time as a single vector 
$$W_t = [w_t, w_{t-1}, ...]^\top$$
then the gradient update rule can be written as some linear equation: 
$$W_{t+1} = CW_t + \alpha \eta e_1$$
where $C$ is some suitable matrix, and $e_1$ is one-hot vector with first entry being 1. Convergence of gradient descent would hence depend on $C$'s eigenvalues, or equivalently, the roots of the characteristic polynomial 
$p(x) = x^{\text{delay}+1} - x^{\text{delay}} + \alpha \lambda$, to lie in the unit circle; solve for an appropriate $\alpha$ gives us a learning rate that lead to convergence. Specifically, [CITE] shows that longer delays require smaller step sizes to ensure convergence. 

For 2, we once again locally approximate the objective function with $f(x) \approx \frac{\lambda}{2}x^2$ for simplicity; then we can can approximate the gradient update equation as 
$$w^+ = w - \nabla f(w_{older}, w_{newer}) \approx w - \lambda w_{newer} - \Delta (w_{newer} - w_{older}) + \eta $$
where $\eta$ denotes a noise term and $\Delta$ is the sensitivity of $\nabla f$ to the discrepancy [todo] [define] of gradient. Now we mimic the strategy shown earlier, express the gradient update equation as a linear equation, and solve for appropriate parameters to make its eigenvalues stay in the unit ball. 

### Zero-Bubble: An Improved Synchronous Approach



## Comparisons and Trade-offs

Due to limited space, the previous section only discussed a few of these approaches in detail. However, if we broadly examine pipeline parallelism methods and their performance metrics in e.g. memory usage, computation resource utilization, and convergence, we see some interesting trade-offs: 

{% include figure.html path="assets/img/comp_table_placeholder.png" class="img-fluid" %}

(will construct our own version of this table)

[CITE]

| Approach      | Schedule   | Bubble Ratio        | Convergence | Extra Mem | Extra Comp | Extra Comm |  
| ------------- | ---------- | ------------------- | ----------- | --------- | ---------- | ---------- |
| GPipe         | Synch      | $\frac{D}{D+T}$     |  Excellent  |           |
| GEMS          | Synch      |  $1 - \Theta(1/D)$  |  Excellent  | X         |
| DAPPLE        | Synch      |  $\frac{D}{D+T}$    |  Excellent  |           |
| Chimera       | Synch      |  $\frac{D}{D+2T}$   |  Excellent  | X         |
| Megatron-LM   | Synch      | $\frac{D}{vT}$      |  Excellent  |           |
| ZeroBubble    | Synch      | $0$                 |  Excellent  |           |
| AMPNet        | Async      | $0$                 |  Poor       |           |
| PipeDream     | Async      | $0$                 |  Good       | X         |    
| XPipe         | Async      | $0$                 |  Good       | X         |    
| SpecTrain     | Async      | $0$                 |  Good       | X         |    
| PipeDream-2BW | Async      | $0$                 |  Good       | X         |    
| PipeMare      | Async      | $0$                 |  Good       | X         |    
| AvgPipe       | Async      | $0$                 |  Good       | X         |    
| WPipe         | Async      | $0$                 |  Good       | X         |    

A survey of these approaches (cite) showed... (add explanation)

[TODO] implicit compute cost: LR schedule constraint. 

<!-- Compute extra memory, compute, and comms -->


<!-- ## Modeling and Optimization

### Problem Formulation

### Proposed Solution: General Purpose Solvers
 -->


---
## TEMPLATE: 
## Equations

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) 
that brought a significant improvement to the loading and rendering speed, which is now 
[on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).


## Images and Figures

Its generally a better idea to avoid linking to images hosted elsewhere - links can break and you
might face losing important information in your blog post.
To include images in your submission in this way, you must do something like the following:

```markdown
{% raw %}{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}{% endraw %}
```

which results in the following image:

{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}

To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2025-04-28-[SUBMISSION NAME]` within your submission.

Please avoid using the direct markdown method of embedding images; they may not be properly resized.
Some more complex ways to load images (note the different styles of the shapes/shadows):

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/8.jpg" class="img-fluid z-depth-2" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/10.jpg" class="img-fluid z-depth-2" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/11.jpg" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/12.jpg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/7.jpg" class="img-fluid" %}
    </div>
</div>

### Interactive Figures

Here's how you could embed interactive figures that have been exported as HTML files.
Note that we will be using plotly for this demo, but anything built off of HTML should work
(**no extra javascript is allowed!**).
All that's required is for you to export your figure into HTML format, and make sure that the file
exists in the `assets/html/[SUBMISSION NAME]/` directory in this repository's root directory.
To embed it into any page, simply insert the following code anywhere into your page.

```markdown
{% raw %}{% include [FIGURE_NAME].html %}{% endraw %} 
```

For example, the following code can be used to generate the figure underneath it.

```python
import pandas as pd
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')

fig = px.density_mapbox(
    df, lat='Latitude', lon='Longitude', z='Magnitude', radius=10,
    center=dict(lat=0, lon=180), zoom=0, mapbox_style="stamen-terrain")
fig.show()

fig.write_html('./assets/html/2025-04-28-distill-example/plotly_demo_1.html')
```

And then include it with the following:

```html
{% raw %}<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>{% endraw %}
```

Voila!

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

***

## Code Blocks

This theme implements a built-in Jekyll feature, the use of Rouge, for syntax highlighting.
It supports more than 100 languages.
This example is in C++.
All you have to do is wrap your code in a liquid tag:

{% raw  %}
{% highlight c++ linenos %}  <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers. You can try toggling it on or off yourself below:

{% highlight c++ %}

int main(int argc, char const \*argv[])
{
string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}

{% endhighlight %}

***

## Diagrams

This theme supports generating various diagrams from a text description using [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} plugin.
Below, we generate a few examples of such diagrams using languages such as [mermaid](https://mermaid-js.github.io/mermaid/){:target="\_blank"}, [plantuml](https://plantuml.com/){:target="\_blank"}, [vega-lite](https://vega.github.io/vega-lite/){:target="\_blank"}, etc.

**Note:** different diagram-generation packages require external dependencies to be installed on your machine.
Also, be mindful of that because of diagram generation the first time you build your Jekyll website after adding new diagrams will be SLOW.
For any other details, please refer to [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} README.

**Note:** This is not supported for local rendering! 

The diagram below was generated by the following code:

{% raw %}
```
{% mermaid %}
sequenceDiagram
    participant John
    participant Alice
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
{% endmermaid %}
```
{% endraw %}

{% mermaid %}
sequenceDiagram
participant John
participant Alice
Alice->>John: Hello John, how are you?
John-->>Alice: Great!
{% endmermaid %}

***

## Tweets

An example of displaying a tweet:
{% twitter https://twitter.com/rubygems/status/518821243320287232 %}

An example of pulling from a timeline:
{% twitter https://twitter.com/jekyllrb maxwidth=500 limit=3 %}

For more details on using the plugin visit: [jekyll-twitter-plugin](https://github.com/rob-murray/jekyll-twitter-plugin)

***

## Blockquotes

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>

***


## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body`-sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

***

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behavior, where trailing spaces are not required.)

* Unordered lists can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links. 
http://www.example.com or <http://www.example.com> and sometimes 
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```
 
```python
s = "Python syntax highlighting"
print(s)
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote. 


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.

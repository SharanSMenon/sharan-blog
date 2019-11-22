---
title: Derivative rules
date: 2019-11-21T17:28:48.740Z
description: This post teaches about the derivative rules with a bonus
---
# Derivative Rules

Hello! In this post, we will learn some of the rules for finding derivatives. This will be very useful when you are trying to find the derivative of a function as this will simplify your work a lot.

## Why do we need these rules

"Why do we need these rules?" I hear you asking. Ok, let me explain. Take the limit definition of the derivative:


$$
\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

Let's try to find the derivative of the following function:
$$
f(x)=\sqrt{2x^3+4x}
$$
Would we really want to plug this function into the limit definition and evaluate the limit? Well, I wouldn't. This is why we have the derivative rules. It simplifies life by a lot. As you will see later on, taking the derivative of this function becomes really easy.

## Table of rules

The following table lists out the 4 main derivative rules that you will use to find derivatives. These rules will simplify your life a lot and they are all easy to learn.
|Rule Name|Rule|
|---|---|
|Power Rule|if $f(x)=x^n$, where $n$ is constant, then $f'(x)=nx^{n-1}$|
|Product Rule|$\frac{d}{dx}(f(x)g(x))=f'(x)g(x) + g'(x)f(x)$|
|Quotient Rule|$\frac{d}{dx}(\frac{f(x)}{g(x)})=\frac{f'(x)g(x) - g'(x)f(x)}{(g(x))^2}$|
|Chain Rule|$\frac{d}{dx}(f(g(x)))=f'(g(x))g'(x)$|
These rules can be used to calculate the derivative of a function. It is a lot easier 

## Examples

Here are some examples of the derivative rules in action

1. Find the derivative of $8x^2$
   The power rule will be used
   $n=2$
   The answer is $16x$
2. Find the derivative of $(x^2+1)*(4x^4+2)$
   Use the product rule
   $f(x)=(x^2+1)$
   $g(x)=(4x^4+2)$
   $f'(x)=2x$
   $g'(x)=16x^3$
   Answer is $2x(4x^4+2) +16x^3(x^2+1)$
   This can be simplified and this should be simplified

### Finding the derivative of the function mentioned earlier

Remember the function $f(x)=\sqrt{2x^3+4x}$? With our newfound rules, let us try taking the derivative of this function. We will use the following rules for this:

1. Chain rule
2. Product rule

For the chain rule, $f(x)=\sqrt{x}$ and $g(x)=2x^3+4x$

The derivative of $g(x)$ is $6x^2 + 4$

The derivative of $f(x)$ is $\frac{1}{2\sqrt{x}}$

Simplifying:
$$
\frac{1}{2\sqrt{2x^3+4x}}(6x^2 + 4)=\frac{3x^2 + 2}{\sqrt{2x^3+4x}}
$$
There we have it. That is the derivative of our function. We used the derivative rules in order to find the derivative of the function.

> **Note**: When you are solving problems that involve derivatives, you typically have to use a combination of the rules like we saw here. Just make sure to keep that in mind.

## Bonus

Here is a little bonus section for you. You need to know Python Programming to follow along in this bonus section.

In this section, we will create a python program to calculate the derivative of a function at some $x$. So this program finds the slope of the tangent line at $x$. We will use the limit definition for this program

### Making the program

Remember the limit definition?

$$
\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

> Ensure that you are on python 3, this program will work only on python 3. Also, the function used in this program is $x^2$. Derivative is $2x$

We will use this since it will be easy to use for this program. Open a python file in your favourite editor and paste the following code in.

```python
from math import  *
import timeit
f =  lambda  x: x**2 # Replace this with your function in python notation
def numDeriv(function,  value):
    """
    Calculates the derivative by using the limit 
    definition. It returns a number that represents 
    the slope of the line tangent to that point.
    """
    h =  0.00000000001
    top =  function(value + h)  -  function(value)
    bottom = h
    slope = top / bottom
    # Returns the slope to the third decimal
    return  float("%.3f"  % slope)
d = numDeriv(f, 4)
print("The slope of the tangent line at x = 4 is {}".format(d)).
```

There you go. That is a fully functional program. Just run it in the terminal and you should see a result.

Well, folks, that's it for this post and thank you for reading and I will see you next time. Bye!

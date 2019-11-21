---
title: Trapezoidal Rule in python
date: 2019-11-21T17:28:48.740Z
description: Implementing the Trapezoidal Rule in Python (and a bonus)
---
Hello! In this tutorial, I am going to show you how to write a python program to implement the trapezoidal rule

Let's get started!

If you want,  [here](https://gist.github.com/SharanSMenon/ccc75cdd28b4fb11e568a64309deecc7) is the link to the Github Gist that contains all the code.

## Prerequisites

There are some things you need to know to follow along with this tutorial. You need a basic knowledge of calculus and you need to know Python Programming. The calculus portion is explained in the next section but this tutorial does not teach python programming. It is recommended that you know calculus so that it will be much easier to follow along and understand. 

## What is the trapezoidal rule?

> _Note_: If you already know what the trapezoidal rule is and how to do integration, then you can skip this section and go on to the part where we write the python program.

This program is going to do integration by using the Trapezoidal rule, but wait. What is the trapezoidal rule?

The trapezoidal rule allows you to calculate the area under a curve. Let's say you have a weird curve and you want to find the area under the curve between $b$ and $a$, maybe it represents something, you would need this integral:

$$\int^b_a f(x) dx$$

$b$ has to be greater than $a$

But how to calculate this integral? That's where the trapezoidal rule comes in. We will start with rectangles and go to trapezoids soon. 

Imagine that you put 5 equal width rectangles to fill up the area that you want to calculate. If you sum up the area of all 5 rectangles, you will get a good approximation of the area under the curve. These are called Riemann sums: Here is the formula

$$\sum\limits_{k=1}^n f(c_k) x_k$$

Where $x_k$ is the width of the interval. Now, let's replace the rectangles with trapezoids, and here is what we get:

$$\sum\limits_{k=1}^n \frac{f(x_{k-1}) + f(x_k)}{2}x_k$$

The Trapezoidal rule is much more accurate than Riemann sums, and as you will see, it is faster when implementing it in a program.

## Making the python program

Ok. Now that we know what the trapezoidal rule is, we can get to make the program. Create a python file called `integration.py` or whatever name you want.

We will write the function first, then we will test the performance of it.

### Writing the function

Create a function called `integrate_trapezoids` or a name of your choice 3 arguments: `N`, `a`, and `b`. `a` and `b` are going to be the bounds of integration, while `N` is the number of trapezoids that we are going to use.
Here is the current state of the program

```py
import math
def integrate_trapeoids(N,a,b):
	pass
```

Now let's implement the trapezoidal rule!

Remember, this was our function:

$$\sum\limits_{k=1}^n \frac{f(x_{k-1}) + f(x_k)}{2}x_k$$

Add the following code to your function

```python
value =  0 # Initial value
subInt =  float(b - a)  / N # Width of Subinterval
value +=  f(a)/2.0 # The first value in the series sum
```

Now, we can get to the heart of the function. After the first code block, add the following code (we are still in the function):

```python
for n in  range(1, N + 1):
	value +=  f(a + n*subInt) # Adding the area of each trapezoid to the total sum
return value * subInt # Just returning the value times the length of each subinterval
```

And that's it! the function is complete. Here is the full code:

```python
def integrate_trapezoids(N, a, b):
	"""
	Integrates the functions given the bounds b and a, by using the trapezoidal rule. 
	It is faster and more accurate than the rectangular method.
	"""
	if a > b:
		raise Exception("a cannot be greater than b")
	value = 0
	subInt = float(b - a)  / N
	value += f(a)/2.0
	for n in range(1, N +  1):
		value += f(a + n*subInt)
	return value * subInt
```

### Testing the performance (Optional)

This is optional,  but we can test the performance of the function to see how fast it is. To do that, just make sure your program is like this:

```python
import math

import timeit # For performance testing

# Function on the line below

f =  lambda  x:  -3**x # Replace the function seen here with your function
def integrate_trapezoids(N,  a,  b):
	if (a > b):
		raise Exception("a cannot be greater than b")
	value = 0
	subInt = float(b - a)  / N
	value += f(a)/2.0
	# Dividing to separate intervals
	for n in  range(1, N +  1):
	value +=  f(a + n*subInt)
	return value * subInt
def test_performance():
	print("Testing performance with N = 15000, a = 0, b = 10, going 200 times")
	setup = "from __main__ import integrate_trapezoids"
	perf1 = timeit.timeit("integrate_trapezoids(10000, 0, 10)",  number=200,  setup=setup)
	print("integrate_trapezoids takes {} milliseconds to run".format(perf1 *  1000))
	
if __name__ ==  "__main__":
	test_performance()
```

This is the full program and it also allows you to test the performance of the function. Just run the program in the terminal and you will get something similar to this:

```
Testing performance with N = 15000, a = 0, b = 10, going 200 times
integrate_trapezoids takes 608.320924 milliseconds to run
```

## Bonus

These bonus sections will have some fun things for us.

### Implementing the Riemann Sum function

Earlier in this tutorial, I talked about Riemann sums but I never showed how to implement them because this article was about the Trapezoidal rule. In this bonus section, I will show how to implement the Riemann sum. If you remember, here was the Mathematical function for the Riemann Sum:

$$\int^b_a f(x) dx=\sum\limits_{k=1}^n f(c_k) x_k$$

Let's implement this in python. I will show you the function. Add the following code under the `integrate_trapezoids` function.

```python
def integrate_rectangular(N,  a,  b):
	"""
	Integrates the function given the bounds 
	b and a, by using Riemann Sums.
	"""
	if (a > b):
		raise Exception("a cannot be greater than b")
	value =  0
	value2 =  0
	for n in range(1, N +  1):
	    value +=  f(a + ((n - (1/2)) * ((b - a)/ N)))
	    value2 =  ((b - a)  / N)  * value
	return value2
```

This function does integration with Riemann sums. Riemann sums are less accurate and the `integrate_trapezoidal` function is faster.

### Using sympy

There is a library called sympy that is capable of doing calculus so you don't have to write this program every time you want to do some integration. You can just use sympy. It can integrate, find derivatives, and much more. It is also a lot faster and way more powerful than the methods used in this program. I recommend that you go ahead and try it out.

Well, folks, that's it for this post. Have fun with your new functions and I will see you later in the next post. Bye!!

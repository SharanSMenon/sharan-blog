---
title: Matrix Operations
date: 2019-11-26T17:18:35.260Z
description: Basic Matrix Operations
---
In this tutorial, you will learn about some simple matrix operations. They include matrix addition, subtraction, and scalar multiplication

## Introduction of Matrices

What is a matrix? According to Wikipedia...

> In mathematics, a **matrix** (plural **matrices**) is a rectangular array of numbers (symbols or expressions), arranged in rows and columns

It is typically displayed like this, 

$$
\begin{pmatrix}
2 & 3 \\
4 & 6
\end{pmatrix}
$$

This is a $2$ x $2$ matrix (it has 2 rows and 2 columns). A $2$ x $3$ matrix would look like this:

$$
a=\begin{pmatrix}
2 & 3 & 7\\
4 & 6  & 5
\end{pmatrix}
$$

It has 2 rows and 3 columns. To get an element inside a matrix, you write a[1, 2]. This finds the element on the first row and the second column. In our case, it's 3.

## Addition of Matrices

Let's do the first of 3 basic operations: Addition. This is very simple. We will have matrices $a$ and $b$. We want to do $a +b$. To do that, we just add the corresponding elements. Let's take 2 matrices, $a$ and $b$.

$$
a=\begin{pmatrix}
2 & 3 & 7\\
4 & 6  & 5
\end{pmatrix}
$$

$$
b=\begin{pmatrix}
4 & 5 & 8\\
5 & 2  & 2
\end{pmatrix}
$$

Here is the result of the addition:

$$
a+b=\begin{pmatrix}
6 & 8 & 15\\
9 & 8  & 7
\end{pmatrix}
$$

You **must** make sure that the matrices have the **same**  dimension. Both matrices here were $2$ x $3$ matrices. But you cannot add a $2$ x $2$ with a $1$ x $2$ or something else.

## Subtraction of Matrices

Subtraction of matrices are the same as addition except you are subtracting the corresponding elements. Again, the matrices **must** have the same dimension.

$$
a=\begin{pmatrix}
2 & 3 & 7\\
4 & 6  & 5
\end{pmatrix}
$$

$$
b=\begin{pmatrix}
4 & 5 & 8\\
5 & 7  & 6
\end{pmatrix}
$$

Here is the result of the addition:

$$
b-a=\begin{pmatrix}
2 & 2 & 1\\
3 & 1  & 1
\end{pmatrix}
$$

## Scalar Multiplication of Matrices

Scalar multiplication of matrices is different. In this operation, you multiply a matrix by a scalar. To multiply a matrix by a scalar, just multiply all the elements by your number

> A Scalar is a ordinary number like 2 or 8

Let's go through an example.

Given matrix $a$, find $3a$.

$$
a=\begin{pmatrix}
2 & 3 \\
4 & 6
\end{pmatrix}
$$

Lets do the multiplication

$$
3a=3*\begin{pmatrix}
2 & 3 \\
4 & 6
\end{pmatrix}
$$

We multiply all elements by $3$. 

$$
3a=\begin{pmatrix}
6 & 9 \\
12 & 18
\end{pmatrix}
$$

And that's it! We just multiplied a matrix by a scalar.

## Conclusion

We have learnt the basic matrix operations, you can now use them wherever you want to. This will be very useful as you go into higher mathematics.

---
title: Introduction to Assembly Language
date: 2022-05-14T15:28:37.456Z
description: >-
  This article will introduce you to the ARM-V8 Assembly Language. Learn to code
  Hello World in Assembly.
---
> **Important**: This tutorial assumes you have an ARM-V**8** device that you can code in (e.g Raspberry Pi 4). This code tutorial was *specifically* written for the Apple M1 family of chips. To run it on another ARM-V8 device, the code might have to be slightly modified. This  code will not work on an ARM-V7 or lower device, because they do not support 64-bit registers.

## What is assembly?

Assembly is a low level programming language where the code very closely resembles machine code instructions. It is an abstraction of machine code so that programmers don't have to go and count 1's and 0's. Each line contains a single statement, which represents a single CPU instruction, but also contain other things like constants, comments, registers, etc.

Assembly programs are compiled using an assembler and then converted into a executable file. In fact, high-level languages like C and C++ compile into Assembly code before turning into an executable.

Assembly languages are specific to different CPU Architectures. For example, ARM chips (RISC) and Intel chips (CISC) use very different assembly languages. Intel chips tend to have more complex instructions that combine multiple instructions into one, while ARM chips have a simpler set of instructions. Even different versions of the same architecture can have different instruction sets. ARM-V7 and ARM-V8 have different instruction sets, thanks to the latter's support of 64-bit computing.

Assembly code is written in `.asm` or `.s` files. Assembly languages give us access to general-purpose registers, which are like a very tiny amount of on-CPU memory, for quick operations. We can also manage stack and heap memory from Assembly, like we do in C/C++. 

## Setup (on macOS)

To get started on macOS, install Xcode and the [Xcode command line tools](https://developer.apple.com/xcode/resources/). The command line tools will provide us with access to the C/C++ compiler (`clang`), the assembler (`as`), and the linker (`ld`). 

> On devices like the Raspberry pi, tools like GCC, the assembler, and the linker should be installed automatically, if you are using Raspbian.

## Hello World Quickstart

Create a file called `HelloWorld.s`. In that file, copy paste the following lines. 

> Different assembly languages have different ways of creating comments. On an Apple M1, // works fine for comments. 

```asm
.global _start			// Provide program starting address to linker
.align 2			// Make sure everything is aligned properly

// Setup the parameters to print hello world
// and then call the Kernel to do it.
_start: mov	X0, #1		// 1 = StdOut
	adr	X1, helloworld 	// string to print
	mov	X2, #14	    	// length of our string
	mov	X16, #4		// Unix write system call
	svc	#0x80		// Call kernel to output the string

// Setup the parameters to exit the program
// and then call the kernel to do it.
	mov     X0, #0		// Use 0 return code
	mov     X16, #1		// System call number 1 terminates this program
	svc     #0x80		// Call kernel to terminate the program

helloworld:      .ascii  "Hello Worldo!\n"
```

We are going to create a `Makefile` to run this code.

```make
HelloWorld: HelloWorld.o
    ld -o HelloWorld HelloWorld.o -lSystem -syslibroot `xcrun -sdk macosx --show-sdk-path` -e _start -arch arm64

HelloWorld.o: HelloWorld.s
    as -o HelloWorld.o HelloWorld.s
```

Now, we can run this code by running the following commands

```sh
$ make -B
$ ./HelloWorld
```

If this works properly, you should get the following message in your terminal

```sh
Hello Worldo!
```

In the next few articles, we will dive more deeper into assembly commands, general-purpose registers, and learn how to make more complicated programs in assembly (e.g for-loops, if statements, etc.)

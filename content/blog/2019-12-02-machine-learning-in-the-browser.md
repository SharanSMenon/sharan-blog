---
title: Machine Learning in the Browser
date: 2019-12-02T16:10:00.327Z
description: Using Tensorflow.JS for machine learning
---
Machine learning is a very powerful tool that can be used for good and bad. Many sites on the web and companies use machine learning nowadays. Machine learning can be used to analyze data, make predictions, classify things, improve user experience, and many more. There are many great applications for machine learning.

One tool that is used to do machine learning is called Tensorflow. We will be using the JavaScript version of Tensorflow which is called Tensorflow.js. So that means we will be doing machine learning in the browser.

## What are we making?

So we know that we are going to use machine learning in the browser, but what exactly are we making?

We will be making a really simple but interesting model to demonstrate the power of neural networks.

Take the equation, $y=2x + 1$. Let's say that we want to make a model that can predict the $y$ given a $x$. We will feed it 5 x-values and 5 y-values. Here is a table showing all the values

| x   | y   |
| --- | --- |
| 0   | 1   |
| 1   | 3   |
| 2   | 5   |
| 3   | 7   |
| 4   | 9   |

This equation describes all of the odd numbers. Our neural network will predict y given an x. So let's say we give the number 54 as $x$. The network will need to predict $y$. We know that $y$ is 108 but the neural network has to predict it based on the numbers that it was given

### How are we building the neural network?

We know what we are building now but how will we build it? Here is the plan

1. Load in tensorflow.js and all other required libraries
2. Get the x and y data
3. Create a neural network
4. Feed the x and y data to the network
5. We have a fully functioning neural network, so we can go ahead and use it to make predictions

The neural network is trying to find a relationship between the x and the y data (in our case, it is trying to figure out what equation works). 

Let's build the network!

## Building the network

Now that we know how what we are building and how we are building it, let's build it.

> **Note**: Make sure that you have Node.js and npm installed before continuing with this tutorial. If you don't, then go ahead and install them.

To get started, create a directory in which your project will live. `cd` into that directory and run the command `npm init -y` in your terminal (make sure that you are in the project directory). In your project directory, create two files. 

### Installing the packages

Let's install all the necessary packages before going on. In your terminal, run the following command:

```
$ npm install @tensorflow/tfjs bootstrap
```

Bootstrap is just to style the application. After that command is finished, run this command next:

```
$ npm install --save-dev parcel
```

> Quick note:  If you have parcel installed globally, then you don't need to run this command.

This installs a bundler called parcel. It is like webpack but we don't have to configure it. It allows us to use ES6 and other cool javascript features.

### Making the program

Now that we have all the packages installed, we can make the program.

In your `index.html` file, copy/paste the following code:

```html
<html>
<body>
    <div class="container" style="padding-top: 20px">
        <div class="card">
            <div  class="card-header">
                <strong>TensorFlow.js Demo</strong>
            </div>
            <div  class="card-body">
                <label>Input Value:</label>  
                <input 
	            type="text" 
	            id="inputValue" 
	            class="form-control"
	         />
                 <br>
                 <button 
	             type="button" 
	             class="btn btn-primary" 
	             id="predictButton" 
	             disabled>Model is being trained, please wait ...
                 </button>
                 <br><br>
                 <h4>Result: </span></h4>
                 <h5>
                     <span  class="badge badge-secondary" id="output"></span>
                 </h5>
                 <h4>Actual Answer: </span></h4>
                 <h5>
                     <span class="badge badge-primary" id="actual"></span>
                 </h5>
            </div>
        </div>
    </div>
    <script  src="./index.js"></script>
</body>
</html>
```

Now let's build the `index.js` file. We will start by importing everything needed and some code that runs when the page is loaded

```js
// Imports
import * as tf from '@tensorflow/tfjs'
import 'bootstrap/dist/css/bootstrap.css'
// Some code that runs when the page is loaded
document.getElementById('output').innerText = "Hello World"
document.getElementById('actual').innerText = "Hello World"
```

Let's build the model. Underneath is the code for the model:

```js
// Initializing the model
const  model  = tf.sequential();
// Adding a single layer
model.add(tf.layers.dense({
	units:  1,
	inputShape: [1]
}))
// Compiling the model
model.compile({
	loss:  "meanSquaredError",
	optimizer:  "sgd"
})
```

This creates a neural network and adds a layer. It then compiles the model and prepares it for training. Under the code for the model, add the following code:

```js
const xs  = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1])
const ys  = tf.tensor2d([-1, 1, 3, 5, 7, 9], [6, 1])
```

This code creates two tensors that contain all the x and y data. Now we will train the model. Add the following code underneath all of your current code:

```js
model.fit(xs, ys, {
	epochs:  500
}).then(()  => {
	document.getElementById('predictButton').disabled = false
	document.getElementById('predictButton').innerText = "Predict"
})
```

That is all the code for the neural network. Now just insert this bit of code at the bottom of the file:

```js
document.getElementById('predictButton').addEventListener(
'click', 
(el, ev) => {
    let val = document.getElementById('inputValue').value;
    let n = parseInt(val)
    // Predicting values
    let predTensor = model.predict(tf.tensor2d([n], [1, 1]))
    let ans = predTensor.dataSync()
    // The actual answer
    let actualAns = (2*n) + 1;
    document.getElementById('output').innerText = ans
    document.getElementById('actual').innerText = actualAns
});
```

There we go! All the code has been written and now we just need to run it. To access in the browser, there are some things we need to do

### Accessing in the browser

In your `package.json`, copy the following code to the `scripts` section:

```
dev":  "parcel index.html"
```

Save `package.json` and run `npm run dev` in your terminal. Then head over to http://localhost:1234 and you should see the model being trained and it should allow you to use the model and predict numbers once the model is finished training. 

## Conclusion

All right. In this tutorial, you learned how to build a neural network in tensorflow.js that can predict a simple equation. There is a lot more than you can do with neural networks from image classification to sentiment analysis. You may see this in future tutorials that I make. 

Well, that's it for this tutorial, I hope to see you in the future for now, bye!

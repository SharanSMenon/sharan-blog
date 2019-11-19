---
title: Creating a GraphQL project
date: 2019-11-12T01:26:01.653Z
description: A project setup for GraphQL
---
GraphQL is a powerful query language for API's developed by the team at Facebook. There are a variety of tools that you can use to create a GraphQL API. We will use [Apollo Server](https://www.apollographql.com/) in this tutorial. This project will be very simple but you can easily expand it. You will learn the basics of GraphQL.

## Prerequisites

You should have a basic understanding of Javascript and GraphQL. You are also going to need some tools:

1. NodeJS (make sure that you are version 8 and above)
2. npm (If you install NodeJS, npm should automatically come along)
3. A Browser (like Google Chrome)
4. A computer (I don't know how you are doing this tutorial without a computer, unless you are just reading. Mac and Linux are good for development, but Windows is more than okay)
5. A Text editor (like Visual Studio Code or WebStorm. I guess you could use Notepad, but it wouldn't be very fun).

> Note: You have to pay for WebStorm, but there is a way to get it for free. If you download the Early Access Program (EAP), you get WebStorm for free. I would recommend VSCode though if you are looking for a free editor.

If you do not have any of these tools, please install (or buy) them before moving on.

## Setting up the project

Make a new folder for your project, I will call mine `graphql-intro`. You can open it in your favorite editor and use the integrated terminal or you can use your terminal. `cd` into the directory and run `npm init`. This command initializes [npm](<>) in your project. 

If you dont want npm to ask you questions, add the `-y` flag. That will skip all the questions.

Now that we have npm installed, we can create a few files.

### Installing Dependencies

Now, lets install all the dependencies from npm. Run the following command below.

```
npm install apollo-server graphql
```

This installs Apollo server and a library from facebook for graphql

We will also use [Nodemon](https://nodemon.io/) to make development easier. To install nodemon, run the following command:

```
npm install nodemon --save-dev
```

### Adding ES6 into the project

We want to use ES6 in this so we will use [Babel](<>). To set up babel, we need to install it first

Run the following command to install babel and its dependencies:

```
npm install @babel/core @babel/node @babel/cli @babel/preset-env --save-dev
```

> This is optional. If you don't want ES6, you can skip this section. Just keep in mind, some of the code in this tutorial might not work. You are going to have to find workarounds

ES6 allows for some cool new syntax, like this:

```js
const express = require('express'); // Normal
import express from 'express'; // ES6

// ES6 Allows for arrow functions
let add = (a, b) => a + b;
// Here is the same function in ES5
function add(a, b) {
	return a + b;
}
```

As you can see, having ES6 can be really nice.

#### Setting up Babel

Now that we have all dependencies installed, we can go ahead and set up Babel

If you know how to set up babel, you can skip this section.

To get started, create a file called `.babelrc`

Copy the following into your `.babelrc`

```json
{
	"presets":["@babel/env"]
}
```

This should be enough for now. You can always learn more at the [Babel](https://babeljs.io/) documentation

### Creating `package.json` scripts

We will create a few npm scripts in `package.json` to make the development process a little easier

So open up `package.json` and just paste the following into the `scripts` section.

```json
{
	"dev":"nodemon src/index.js --exec babel-node"
}
```

### Creating the project files

In your project folder, make a directory called `src` and a file called `index.js` inside that `src` directory

## Building the server

Finally, we finished all the setup needed to start this project. We can now write some code. Lets get started!

We are going to build a very simple server that shows off the basics of GraphQL.
Open `index.js` and paste the following in.

```js
const { ApolloServer, gql } = require('apollo-server');

// The GraphQL schema
const typeDefs = gql`
 type Query {
 "A simple type for getting started!"
 hello: String
 }
`;

// A map of functions which return data for the schema.
const resolvers = {
  Query: {
    hello: () => 'world',
  },
};

const server = new ApolloServer({
  typeDefs,
  resolvers,
});

server.listen().then(({ url }) => {
  console.log(`Server ready at ${url}`);
});
```

And there we are. This is a fully functional but simple server that is easily expandable.
Type in `npm run dev` to the terminal to start the server

## Conclusion

In this tutorial, we learned how to build a GraphQL Server using Apollo Server. You can easily expand this project due to its flexible file structure.

Here are some bonus ideas to get you started.

### Bonus Ideas

1. Deploy this app to production (using Heroku or something else)
2. Add a database (SQLite, MongoDB, anything). You can use Sequelize or Mongoose depending on which database you choose (See below for more)
3. Add Authentication (passport.js, or custom authentication with jwt's and bcrypt)
4. Install a test runner like Jest and make some tests for your server
5. Create a client to interact with your server (Angular, React, etc.)
6. Move the schema to its separate file as it gets more complex
   ### Bonus App Ideas
   Here are some ideas for an app you can build the backend with GraphQL
7. A Blog App
8. A Todo App (Classic)
9. A Expenses Manager
10. A Lists manager (It manages a list of lists)
11. A Notes app
12. A Recipes App

There are tons of applications that you can build. For all these applications, you can build a backend with GraphQL.

> **Databases**: If you want to use a database (which you should), here are some options
> You first have to just add a folder called `db` to src where database connection is managed (Just staying organized)
>
> 1. MySQL
> 2. MongoDB
> 3. SQLite
> 4. PostgreSQL
> 5. MariaDB
>
> You also probably want a libarary to use the database. You can use the native database driver for your database, but you can use another solution like:
>
> 1. Mongoose (MongoDB Only) (JS)
> 2. Sequelize (SQL Databases) (JS)
> 3. SQL Alchemy (SQL Databases) (Python)
> 4. Prisma (Many databases) (JS)
>
> **Quick Note**: If you are building a frontend and don't want to style your application with CSS, try these alternatives:
>
> 1. Material Design
> 2. Bootstrap
> 3. Semantic UI

## More Resources

If you are really interested in GraphQL. I recommend that you take a course on GraphQL. There are many courses online. Here are a few

1. [Complete GraphQL Bootcamp](<>)
   Thank you for reading this tutorial and I hope you build something great.

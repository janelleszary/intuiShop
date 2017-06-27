#IntuiShop

According to a 2016 Pew Research Center study <http://www.pewinternet.org/2016/12/19/online-shopping-and-e-commerce/> nearly 80% of Americans shop online. But, all else being equal, the study also shows that most consumers *prefer to shop at brick-and-mortar stores*. How can e-commerce retailers make a better shopping experience for consumers? There are certainly many factors which contribute to the customer experience, but here I detail a project completed as part of a consultation project for a company which uses proprietary AI software to help users visually search for products. Think reverse-image-search for clothing... or a Pinterest board which actually shows you how to buy the things you like!

Finally, an exploratory extension of the project integrates image search with semantic text. This product would allow a user query such as "I like this dress [urlofimage.jpg] but want it to be more *formal*". While keyword-based search is already possible (filter for color = yellow), this search would be more semantic, allowing you to use intuitive terms to find products that meet fuzzy, semantic criteria. Now *that* would be intuitive shopping!

This project was completed in 3 weeks as part of the Insight Data Science Fellowship program in New York, NY, in June 2017.


###Challenge

My client is currently using a computer vision search algorithm to match fashion products to items in their database. In order to improve their product (the search algorithm), the company has two data science questions:
``1.`` Our pipeline currently includes visual information as the basis for categorization decisions, but how important is text information (i.e. the descriptions of clothing products)? That is, how much better would a classifier do if using text information? Text + image information?

``2.`` The items in our database are currently organized according to hand-generated categories that seemed to make sense to us. How would those categories change if we used a data-driven approach? What kind of latent category structure is in the data?


###Solution

*[A more technical deep-dive detailing this solution is provided below.]*

In order to answer the client's questions, I used neural networks to generate high dimensional vector representations of (a) product images, (b) the semantics of product text descriptions, and (c) combinations of image + text.

``1.`` To answer the first question, I used a variety of supervised learning methods to test the degree to which the different sources of information contained category information. That is, given the image vector, can a supervised learning model accurately decide whether an item is a sweater or a blazer? Could it do so with the image vector?

``2.`` To answer the client's second question, I applied unsupervised learning methods to the collection of vectors (temporarily ignoring the fact that they already had category labels) to see what category structures emerged naturally from the data (when using the same number of categories). Then, I compared the emergent labels to the existing labels in order to get a sense of which categories could be redefined in order to make the categories more separable.

``3.`` With vectors representing the text and image features of each item, we can think of the product database as an abstract, high dimensional space. Each object can be projected into a location in this space based on the values of its image and text features. I realized that we could also do this in reverse -- take a location in space, and go backwards to see what objects are most similar. So, if we start with one product (say a particular Zac Posen dress) we can essentially change the text embedding (using user-defined input terms such as "fancy" or "summery") and locate the new, hypothesized item's location in product-space. Then, we can see what products are nearby and return those to the user.


###Result

``1.`` Several supervised learning methods were tested, but multiclass logistic regression (LR) proved to be the most successful overall. Using LR and image features, products were correctly classified 84.1% of the time. Using text features, products were correctly classified 84.5% of the time. Finally, using both features, products were classified correctly 91.2% of the time. Thus, I reported to my client that while classification is already fairly good (using just images, as is currently done), it could be improved by 8% if text features are integrated.

``2.`` A confusion matrix plotting the results of the above classification task gave an initial indication of which categories were more or less distinct. Combining this with visual comparison of the unsupervised learning algorithm, I was able to suggest two categories that should be combined and two that could be created in order to define more separable categories. With more separable categories, the client's AI will have an easier time labeling new objects.

``3.`` The product does return the most similar items, but because the dimensionality is already so high, changes to the word vectors don't have much impact. That is, there are only slight (but interesting) differences in the objects returned for my Zac Posen dress + "fancy" query versus my Zac Posen dress + "casual" query. So I'm not going to be applying for start-up funding any time soon, but had fun playing with the idea! 


##Technical deep-dive

**Coming soon:** A detailed description of how this project was executed--how the data was collected, cleaned and sampled; how the algorithms and parameters were chosen; how the results were interpreted.


##How to use this repository
**Coming soon.**

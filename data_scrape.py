import nltk
from newspaper import Article
import newspaper
from newspaper import fulltext 
import requests

print("Scrape a website may take a while........")
cnn_article=newspaper.build("https://www.nytimes.com/section/world",memoize_articles=False)

count=1
for article in cnn_article.articles:
    article.download()
    article.parse()
    article.nlp()
    with open("article_title.txt","a")as f,open("article_summary.txt","a")as su,open("article_keywords.txt","a") as ke:
        f.write("\n")
        f.write("Article:")
        f.write(str(count))
        f.write("\t\t")
        f.write(article.title)
        su.write("\n")
        su.write("Summary:")
        su.write(str(count))
        su.write("\t\t")
        su.write(article.summary)
        su.write("\n") 
        ke.write("\n")
        ke.write("Keywords:")
        ke.write(str(count))
        ke.write("\t\t")
        ke.write(str(article.keywords))
        ke.write("\n") 
    print(article.title)
    print("\n")
    print(article.summary)
    print(article.keywords)
    print("*****************************")
    count+=1


   

















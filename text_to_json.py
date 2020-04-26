import json

Article_det={}
with open("article_title.txt","r") as t ,open("article_summary.txt","r") as s,open("article_keywords.txt","r") as k:
    for line in enumerate(t.readlines()):   
        Article_det["Article"] = str(line).strip()
    for line in enumerate(s.readlines()):  
        Article_det["Summary"]=str(line).strip()
    for line in enumerate(k.readlines()):
        Article_det["keywords"]=str(line).strip()        
print(Article_det)
json=json.dumps(Article_det)
json_file=open("Article_details.json","w")
json_file.write(json)
json_file.close()  
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800" #books封面大小
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(), #封面无的时候，其他时候就是url
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()  #
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0.0000000000000001, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k) #提取 ISBN → 回到 books 表（拿到完整信息）
    # #输入用户查询 query，在向量数据库里做语义检索，先取 initial_top_k=50。
    #（比如用户输入 "forgiveness"，就找到“宽恕”相关的书描述）。
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    #个 Document 其实是 “ISBN + 描述” 拼起来的。 这行代码就是从每个结果里提取出 ISBN号，方便后面回到原始 books 表里查
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
    #根据 ISBN 去 books DataFrame 里找出完整的书籍信息（标题、作者、情绪分数

    if category != "All": #分类过滤（只看用户想要的类）
        #如果用户选了分类（比如 Fiction）：只保留这个分类里的结果
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy": #情绪排序（个性化推荐）
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
        #Happy → 按 joy 分数从高到低
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    #调用 retrieve_semantic_recommendations 从向量数据库里找到推荐书籍（一个 DataFrame）。

    for _, row in recommendations.iterrows(): #把 书籍简介（description） 截断为前 30 个单词，避免太长。
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}" #生成展示的文字（caption
        results.append((row["large_thumbnail"], caption))
    return results

# 分类和情绪选项
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

#新建一个 Gradio dashboard，用 Glass 主题，美观半透明效果。 顶部显示标题
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",  #Textbox：用户输入描述
                                placeholder = "ivy's project")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All") #Dropdown (Category)：选择分类
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All") #Dropdown (Tone)：选择情绪基调
        submit_button = gr.Button("Find recommendations") #Button：点击后执行推荐

    gr.Markdown("## Recommendations") #输出控件
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,  #当用户点击按钮 → 调用 recommend_books
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)
    #输入：用户输入的 query + 分类 + 情绪 。输出：结果传给 Gallery 显示


if __name__ == "__main__":
    dashboard.launch()
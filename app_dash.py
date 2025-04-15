import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import re
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# === Load and clean dataset ===
df = pd.read_csv("realistic_100k_comments.csv")
df = df.dropna(subset=['body', 'sentiment', 'score'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text

# === Extract word stats for treemap ===
topics = {
    "climate change": ["climate", "change"],
    "global warming": ["global", "warming"],
    "bernie": ["bernie"],
    "energy": ["energy"],
    "world": ["world"]
}
stopwords = ENGLISH_STOP_WORDS.union({
    "gt", "lt", "amp", "st", "ve", "ll", "re", "im", "dont",
    "didnt", "doesnt", "don", "just", "people"
})
all_keywords = set(word for lst in topics.values() for word in lst)

records = []
for topic, keywords in topics.items():
    pattern = "|".join([rf"\b{k}\b" for k in keywords])
    topic_df = df[df['body'].str.lower().str.contains(pattern, na=False)].copy()
    topic_df["cleaned"] = topic_df["body"].map(clean_text)

    word_stats = defaultdict(list)
    for text, sentiment, score in zip(topic_df["cleaned"], topic_df["sentiment"], topic_df["score"]):
        words = text.split()
        for word in words:
            if word not in stopwords and word not in all_keywords and len(word) > 2:
                word_stats[word].append((sentiment, score))

    top_words = sorted(word_stats.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for word, vals in top_words:
        sentiments, scores = zip(*vals)
        records.append({
            "topic": topic,
            "word": word,
            "count": len(vals),
            "avg_sentiment_score": sum(sentiments) / len(sentiments),
            "avg_comment_score": sum(scores) / len(scores)
        })

df_words = pd.DataFrame(records)

# === Build app ===
app = dash.Dash(__name__)
app.title = "Climate Change Treemap Explorer"

app.layout = html.Div([
    html.H2("🌍 Climate Change Reddit Explorer"),
    html.P("Click on a word in the treemap to explore related Reddit comments."),
    
    html.Label("Color by:"),
    dcc.Dropdown(
        id="color-metric",
        options=[
            {"label": "Count", "value": "count"},
            {"label": "Avg_sentiment_score", "value": "avg_sentiment_score"},
            {"label": "Avg_comment_score", "value": "avg_comment_score"},
        ],
        value="avg_sentiment_score",
        clearable=False,
        style={"width": "300px", "marginBottom": "20px"}
    ),

    dcc.Graph(id="treemap", style={"height": "600px"}),
    html.Div(id="comments-output", style={"marginTop": "30px"})
])

# === Update treemap figure based on selected metric ===
@app.callback(
    Output("treemap", "figure"),
    Input("color-metric", "value")
)
def update_treemap(color_metric):
    colorbar_title = {
        "count": "Count",
        "avg_sentiment_score": "Avg Sentiment",
        "avg_comment_score": "Avg Comment Score"
    }[color_metric]

    color_scale = {
        "count": "Blues",
        "avg_sentiment_score": "RdYlGn",
        "avg_comment_score": "Purples"
    }[color_metric]

    fig = px.treemap(
        df_words,
        path=["topic", "word"],
        values="count",
        color=color_metric,
        color_continuous_scale=color_scale,
        custom_data=["word", "avg_sentiment_score", "avg_comment_score"]
    )

    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Avg Sentiment: %{customdata[1]:.3f}<br>Avg Score: %{customdata[2]:.2f}<extra></extra>"
    )
    fig.update_coloraxes(colorbar_title=colorbar_title)
    return fig

# === Show top 5 comments for clicked word ===
@app.callback(
    Output("comments-output", "children"),
    Input("treemap", "clickData")
)
def display_comments(clickData):
    if not clickData:
        return html.P("Click on a word to see related comments.")

    selected_word = clickData["points"][0]["label"].lower()
    matches = df[df["body"].str.lower().str.contains(rf"\b{re.escape(selected_word)}\b", na=False)]
    matches = matches[matches["body"].str.len() > 30].drop_duplicates(subset=["body"]).sort_values(by="sentiment", ascending=False)

    if matches.empty:
        return html.P(f"No comments found for '{selected_word}'.")

    top_comments = matches.head(5)

    return html.Div([
        html.H4(f"💬 Top comments for '{selected_word}':"),
        html.Ul([
            html.Li([
                html.Div(row["body"], style={"marginBottom": "5px"}),
                html.Small(f"Sentiment: {row['sentiment']:.2f} | Score: {row['score']}")
            ]) for _, row in top_comments.iterrows()
        ])
    ])

if __name__ == "__main__":
    app.run(debug=True)


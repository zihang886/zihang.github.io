import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import re
from collections import defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

# === Load and clean dataset ===
df = pd.read_csv(
    "https://docs.google.com/spreadsheets/d/1HvxKGtsi1h91f5Zna3zmQ41zijuE9uwOhv6hTDLlaVA/export?format=csv",
    nrows=10000
)
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
        for word in text.split():
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

    html.Div([
        html.Strong("Metric explanation:"),
        html.Ul([
            html.Li("Count: Number of times the word appears in related comments."),
            html.Li("Avg Sentiment Score: Mean sentiment score for comments containing the word (-1 to 1)."),
            html.Li("Avg Comment Score: Mean Reddit upvote score for comments containing the word.")
        ])
    ], style={"marginBottom": "20px"}),

    html.Label("Color by:"),
    dcc.Dropdown(
        id="color-metric",
        options=[
            {"label": "Count", "value": "count"},
            {"label": "Avg Sentiment Score", "value": "avg_sentiment_score"},
            {"label": "Avg Comment Score", "value": "avg_comment_score"},
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
    fig = px.treemap(
        df_words,
        path=["topic", "word"],
        values="count",
        color=color_metric,
        color_continuous_scale={
            "count": "Blues",
            "avg_sentiment_score": "RdYlGn",
            "avg_comment_score": "Purples"
        }[color_metric],
        custom_data=["word", "count", "avg_sentiment_score", "avg_comment_score"]
    )

    fig.update_traces(
        hovertemplate="<b>词语（%{customdata[0]}）</b><br>" +
                      "Count: %{customdata[1]}<br>" +
                      "Avg Sentiment Score: %{customdata[2]:.3f}<br>" +
                      "Avg Comment Score: %{customdata[3]:.2f}<extra></extra>"
    )

    fig.update_coloraxes(colorbar_title={
        "count": "Count",
        "avg_sentiment_score": "Avg Sentiment",
        "avg_comment_score": "Avg Comment Score"
    }[color_metric])
    return fig

# === Show 5 random comments with highlighted word ===
@app.callback(
    Output("comments-output", "children"),
    Input("treemap", "clickData")
)
def display_comments(clickData):
    if not clickData:
        return html.P("Click on a word to see related comments.")

    selected_word = clickData["points"][0]["label"].lower()
    matches = df[df["body"].str.lower().str.contains(rf"\b{re.escape(selected_word)}\b", na=False)]
    matches = matches[matches["body"].str.len() > 30].drop_duplicates(subset=["body"])

    if matches.empty:
        return html.P(f"No comments found for '{selected_word}'.")

    top_comments = matches.sample(n=5, random_state=1) if len(matches) >= 5 else matches

    def highlight(text, word):
        return re.sub(rf"(\b{re.escape(word)}\b)", r"<mark>\g<1></mark>", text, flags=re.IGNORECASE)

    return html.Div([
        html.H4(f"💬 Top comments for '{selected_word}':"),
        html.Ul([
            html.Li([
                dcc.Markdown(highlight(row["body"], selected_word), dangerously_allow_html=True,
                             style={"marginBottom": "5px"}),
                html.Small(f"Sentiment: {row['sentiment']:.2f} | Score: {row['score']}")
            ]) for _, row in top_comments.iterrows()
        ])
    ])

# === Run on server ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8051))
    app.run(host="0.0.0.0", port=port)

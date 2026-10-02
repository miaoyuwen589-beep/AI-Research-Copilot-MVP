# Human Sentiment Labelling Guide

Label each headline using only the information visible in that headline. Do not
look at the later share-price movement and do not change a label after seeing a
model prediction.

## Positive (`P`)

Use when the headline clearly describes a favourable development for Apple,
such as stronger sales or profit, successful product demand, favourable legal
outcomes, improved guidance, or another development likely to strengthen the
company's financial or competitive position.

## Negative (`N`)

Use when the headline clearly describes an adverse development for Apple, such
as weaker sales or profit, product problems, regulatory penalties, lawsuits,
supply disruption, deteriorating guidance, or another material business risk.

## Neutral (`U`)

Use for routine announcements, factual descriptions without a clear direction,
ambiguous wording, or mixed headlines containing both positive and negative
information. When evidence is insufficient, choose neutral rather than guess.

## Consistency rules

1. Label the effect on Apple, not whether the writing sounds optimistic.
2. Do not use Alpha Vantage's sentiment field or FinBERT output.
3. Treat mixed or unclear headlines as neutral.
4. Complete all 30 labels before running the model comparison.
5. Report that one reviewer labelled the sample; this is a limitation.

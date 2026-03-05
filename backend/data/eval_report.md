# Druta Kart AI Evaluation Report

Generated: 2026-03-05 18:42 UTC

## Summary

| Metric | Value |
|--------|-------|
| Conversations tested | 50 |
| Resolution rate | 46.0% |
| Intent accuracy | 16.0% |
| Language accuracy | 86.0% |
| Avg tone score | 4.5 / 5 |
| Hallucination rate | 0.0% |
| Offer compliance | 100.0% |

## Token Usage & Cost

| Model | Tokens | Cost |
|-------|--------|------|
| llama-3.1-8b-instant | 38,560 | $0.0023 |
| llama-3.3-70b-versatile | 42,412 | $0.0250 |
| **Total** | **80,972** | **$0.0273** |

Total API calls: 167
Rate limit hits: 0
Elapsed: 13.4 min

## Scenario Breakdown

| Scenario | Count | Resolution | Intent Acc |
|----------|-------|------------|------------|
| damaged_product      |    10 |   30.0% |   10.0% |
| happy_path           |    11 |   45.5% |   18.2% |
| late_delivery        |     5 |    0.0% |    0.0% |
| missing_item         |    10 |   80.0% |    0.0% |
| payment_issue        |     6 |   50.0% |   83.3% |
| wrong_item           |     8 |   50.0% |    0.0% |

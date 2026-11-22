# Process Log

## 1. AI tools used

Claude (Opus 5.5) in the terminal wrote most of the code from my design and specs. I also
used separate Claude sessions as reviewers: they graded the project against the rubric and
tried to break the live service.

## 2. Development narrative

Assignment 1 lost points because its only model was a paid OpenRouter model. Another process
drained the shared credits, and every graded request returned a 402 error instead of an
answer. For A2 I removed that single point of failure: requests route across three free
providers (NVIDIA, Gemini, OpenRouter), OpenRouter is restricted to free models in code, and a
test forces a 402 to prove a fallback still answers.

Free tiers then became the main constraint. While building and testing I hit Gemini's daily
quota, Tavily's 100-requests-per-minute limit on the dev key, and upstream rate limits on
OpenRouter's free models. Each one changed the design: a failing provider cools down instead
of being retried, web search falls back to DuckDuckGo, and the trace shows which model
actually answered.

## 3. Verification narrative

The 285 offline tests re-ran after every change. I also checked the deployed app by hand with
real questions, and the reviewers ran adversarial queries against it. They found bugs the
tests missed: answers split into three sections because the model copied a "---" separator
from its sources; "ColBERT" retrieved nothing because the search model did not know the name;
and a prompt-injection query made the data agent invent rows. Each fix got its own test.

## 4. What I learned

Documentation describes a free tier; only live calls show its real limits. With free models,
reliability comes from the fallback design, not from any single provider.

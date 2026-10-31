# Eval results: RAG vs agent

- Run: 2026-10-30 21:48 on the Bookshelf sample repo (`eval/sample_repo/`)
- Answer model: `qwen2.5:3b` · Judge model: `qwen2.5:3b` · temperature 0.0
- Questions: 20 (8 single_hop, 8 multi_hop, 4 unanswerable)

## Summary

| Metric | RAG | AGENT |
|---|---|---|
| Correctness (1-5) | 4.65 | 3.35 |
| Faithfulness (1-5) | 3.20 | 2.35 |
| Citation quality (1-5) | 3.20 | 2.65 |
| Overall (1-5) | 3.68 | 2.78 |
| Pass rate (correctness ≥ 4) | 90% | 50% |
| Avg latency (s) | 11.0 | 14.7 |
| Avg tool calls | 0.0 | 1.4 |
| Avg model rounds | 1.0 | 3.0 |
| Runs that hit the iteration limit | 0 | 0 |
| Crashed runs | 0 | 0 |
| Judge replies that couldn't be parsed | 0 | 0 |

## Correctness by question type

| Type | RAG | AGENT |
|---|---|---|
| single_hop | 4.25 | 3.75 |
| multi_hop | 4.88 | 2.62 |
| unanswerable | 5.00 | 4.00 |

## Per question

| ID | Type | Mode | Correct | Faithful | Citations | Tools | Rounds | Latency (s) | Judge's reason |
|---|---|---|---|---|---|---|---|---|---|
| m01 | multi_hop | agent | 1 | 1 | 1 | 2 | 4 | 25.7 | The assistant's answer is incorrect because it assumes a value for `LATE_FEE_PER_DAY` and calculates a late fee based on this assumption, which is not supported |
| m01 | multi_hop | rag | 5 | 4 | 4 | 0 | 1 | 18.8 | The assistant correctly identifies the `calculate_late_fee` function and applies the correct logic to calculate the late fee. The citation is accurate and speci |
| m02 | multi_hop | agent | 3 | 2 | 5 | 2 | 4 | 20.0 | The assistant correctly identified the maximum late fee and the function responsible for calculating it. However, the assistant incorrectly inferred that the co |
| m02 | multi_hop | rag | 5 | 4 | 4 | 0 | 1 | 7.8 | The assistant correctly identified the maximum late fee and the threshold for paying fees. The assistant also accurately described the error raised when attempt |
| m03 | multi_hop | agent | 5 | 4 | 4 | 1 | 2 | 31.9 | The assistant correctly identifies the relevant functions and files from the codebase, and the claims are supported by the retrieved sources. The assistant does |
| m03 | multi_hop | rag | 5 | 5 | 5 | 0 | 1 | 18.0 | The assistant's answer accurately describes the sequence of events and code logic for returning a reserved book, and all claims are supported by the provided co |
| m04 | multi_hop | agent | 3 | 2 | 3 | 1 | 3 | 13.1 | The assistant's answer correctly identifies some conditions from the code that can refuse a loan renewal, such as the loan being returned, overdue, or having re |
| m04 | multi_hop | rag | 5 | 4 | 4 | 0 | 1 | 8.9 | The assistant correctly identifies the reasons a loan renewal can be refused based on the provided code. The assistant also correctly cites the relevant lines f |
| m05 | multi_hop | agent | 4 | 3 | 3 | 1 | 3 | 10.9 | The assistant correctly identified the two API endpoints that are limited to librarians. However, the assistant incorrectly inferred that the role check is enfo |
| m05 | multi_hop | rag | 4 | 3 | 4 | 0 | 1 | 12.0 | The assistant correctly identified the two endpoints that are limited to librarians and provided a clear explanation of how these endpoints are enforced. Howeve |
| m06 | multi_hop | agent | 1 | 1 | 1 | 1 | 2 | 6.6 | The assistant's answer indicates that the codebase does not contain specific details about what happens to logged-in users and library data when the server rest |
| m06 | multi_hop | rag | 5 | 4 | 3 | 0 | 1 | 12.7 | The assistant correctly identifies that the data is cleared when the server restarts and accurately describes the in-memory storage of login tokens and library  |
| m07 | multi_hop | agent | 3 | 3 | 3 | 1 | 3 | 14.6 | The assistant correctly identified the two options a member can take if a book has no copies available: reserve the book or renew an existing loan. However, the |
| m07 | multi_hop | rag | 5 | 4 | 4 | 0 | 1 | 12.5 | The assistant's answer correctly identifies the `reserve_book` function and its behavior, which is supported by the provided code snippets. The assistant accura |
| m08 | multi_hop | agent | 1 | 1 | 1 | 2 | 4 | 16.3 | The assistant's answer is incorrect because it does not correctly identify the relevant section of the code. The `checkout_book` function is actually defined in |
| m08 | multi_hop | rag | 5 | 4 | 4 | 0 | 1 | 29.0 | The assistant correctly traces the checkout request from the HTTP route to the creation of a new loan, and supports all claims with the retrieved sources. The c |
| s01 | single_hop | agent | 5 | 4 | 4 | 3 | 5 | 23.6 | The assistant correctly identifies that the loan period is set to 14 days in `config.py` and that this value is retrieved from the environment variable `BOOKSHE |
| s01 | single_hop | rag | 5 | 4 | 4 | 0 | 1 | 10.3 | The assistant correctly identifies the loan period as 14 days and states that it can be changed by modifying the `config.py` file, which is supported by the pro |
| s02 | single_hop | agent | 5 | 4 | 4 | 1 | 2 | 20.0 | The assistant's answer correctly identifies the algorithm used in `hash_password` and the storage of the salt. The faithfulness score is 4 because while all cla |
| s02 | single_hop | rag | 5 | 4 | 4 | 0 | 1 | 10.4 | The assistant's answer correctly identifies the algorithm used and the method of storing the salt, but it invents a citation for the `hash_password` function, w |
| s03 | single_hop | agent | 3 | 3 | 5 | 2 | 4 | 13.7 | The assistant correctly identifies that the exact value of `TOKEN_TTL_MINUTES` is not provided in the codebase and acknowledges that the specific duration of lo |
| s03 | single_hop | rag | 5 | 4 | 4 | 0 | 1 | 9.2 | The assistant correctly identifies that the login token is valid for `TOKEN_TTL_MINUTES` minutes, which matches the reference answer. The assistant also correct |
| s04 | single_hop | agent | 5 | 5 | 5 | 2 | 4 | 21.0 | The assistant correctly identified that `MAX_ACTIVE_LOANS` is not defined in the `models.py` file and confirmed this by searching for it in the `loans.py` file. |
| s04 | single_hop | rag | 3 | 3 | 3 | 0 | 1 | 11.2 | The assistant's answer correctly identifies the `MAX_ACTIVE_LOANS` constant and its value, but it invents a value for `MAX_ACTIVE_LOANS` (5) instead of using th |
| s05 | single_hop | agent | 5 | 4 | 4 | 1 | 3 | 12.1 | The assistant correctly identified the `MAX_RENEWALS` and `RENEWAL_DAYS` constants from the `config.py` file and accurately stated their values. The assistant a |
| s05 | single_hop | rag | 5 | 4 | 4 | 0 | 1 | 6.1 | The assistant correctly identified the maximum number of renewals and the number of days added per renewal, which is supported by the code and configuration fil |
| s06 | single_hop | agent | 5 | 4 | 4 | 1 | 2 | 13.6 | The assistant's answer correctly describes the `verify_password` function and its process of comparing hashes. It accurately cites the relevant code snippets fr |
| s06 | single_hop | rag | 5 | 4 | 4 | 0 | 1 | 15.8 | The assistant's answer correctly describes the process of verifying passwords and the use of `hmac.compare_digest` to prevent timing attacks. The faithfulness s |
| s07 | single_hop | agent | 1 | 1 | 1 | 1 | 2 | 8.7 | The assistant's answer incorrectly states that the API returns a 404 status code when a borrowing rule is broken. The reference answer clearly states that the A |
| s07 | single_hop | rag | 1 | 1 | 1 | 0 | 1 | 9.2 | The assistant's answer incorrectly states that the API returns a 404 status code when a borrowing rule is broken. The code snippet in `api.py` shows that the AP |
| s08 | single_hop | agent | 1 | 1 | 1 | 2 | 4 | 11.8 | The assistant's answer correctly states that the `notifications.py` file does not contain information about the address for notification emails or whether they  |
| s08 | single_hop | rag | 5 | 4 | 4 | 0 | 1 | 6.1 | The assistant's answer correctly identifies the sender address and accurately describes the behavior in development. The answer is faithful to the provided sour |
| u01 | unanswerable | agent | 5 | 1 | 1 | 1 | 2 | 7.0 | The assistant's answer correctly states that two-factor authentication is not implemented in the codebase and agrees with the reference answer. However, the ans |
| u01 | unanswerable | rag | 5 | 1 | 1 | 0 | 1 | 6.9 | The assistant correctly states that the provided code snippets do not contain any information related to two-factor authentication, which aligns with the refere |
| u02 | unanswerable | agent | 5 | 1 | 1 | 1 | 2 | 7.1 | The assistant correctly states that the provided code snippets do not directly address the rate limits applied to each user and recommends looking for documenta |
| u02 | unanswerable | rag | 5 | 1 | 1 | 0 | 1 | 4.1 | The assistant's answer correctly states that there is no rate limiting in the codebase, matching the reference answer. However, the answer is considered unfaith |
| u03 | unanswerable | agent | 1 | 1 | 1 | 1 | 2 | 11.9 | The assistant's answer does not provide any evidence or claims that contradict the reference answer, which states there is no search endpoint. The assistant's e |
| u03 | unanswerable | rag | 5 | 1 | 1 | 0 | 1 | 5.8 | The assistant correctly states that the provided code does not include an endpoint for searching books by author, matching the reference answer. However, the an |
| u04 | unanswerable | agent | 5 | 1 | 1 | 1 | 2 | 4.1 | The assistant correctly states that the Dockerfile is not found in the codebase, matching the reference answer. However, since the reference answer says there's |
| u04 | unanswerable | rag | 5 | 1 | 1 | 0 | 1 | 4.2 | The assistant correctly states that the provided code snippets do not contain information about the cloud provider or the Dockerfile, matching the reference ans |

## Caveats

- The judge is a small local model. When it is the same model that wrote the answers, scores tend to be generous; set `JUDGE_MODEL` to a different model for a stricter grade.
- 20 questions is a small sample: treat differences under ~0.3 points as noise.
- Latency was measured on the machine that ran the eval and includes the first model load.

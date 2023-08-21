def word_break(s, word_dict):
    """
    Given a string s and a set of words dict,
    determine if s can be segmented into a sequence of words
    where all the words are present in the dict
    (the sequence can include one or more words).
    For example, given s = "hellomsraalgo" and dict = ["hello", "msra", "algo"],
    return true because "hellomsraalgo" can be segmented into "hello msra algo".
    """
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_dict:
                dp[i] = True
                break

    return dp[n]


s = "hellomsraalgo"
word_dict = ["hello", "msra", "algo"]
result = word_break(s, word_dict)
print(result)

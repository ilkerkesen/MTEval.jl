function countdict(xs)
    counts = Dict()
    for x in xs
        if haskey(counts, x)
            counts[x] += 1
        else
            counts[x] = 1
        end
    end
    return counts
end

get_words(sentence) = map(lowercase, split(sentence, " "))
build_ngram(words, n) = map(i -> join(words[i:i+n-1], " "), 1:length(words)-n+1)
ngram_counts(sentence, n) = countdict(build_ngram(get_words(sentence), n))
